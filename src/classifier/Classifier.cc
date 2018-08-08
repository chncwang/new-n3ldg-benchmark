#include "Classifier.h"
#include "Category.h"

#include <chrono> 
#include <unordered_set>
#include "Argument_helper.h"
#include "Reader.h"
#include "profiler.h"
#if USE_GPU
#include "N3LDG_cuda.h"
#endif

Classifier::Classifier(int memsize) : m_driver(memsize) {
    srand(0);
}

Classifier::~Classifier() {}

int Classifier::createAlphabet(const vector<Instance> &vecInsts) {
    if (vecInsts.size() == 0) {
        std::cout << "training set empty" << std::endl;
        return -1;
    }
    std::cout << "Creating Alphabet..." << endl;

    int numInstance;

    for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
        const Instance *pInstance = &vecInsts.at(numInstance);

        vector<const string *> words;
        for (const string &w : pInstance->m_title_words) {
            words.push_back(&w);
        }

        for (const string *w : words) {
            string normalizedWord = normalize_to_lowerwithdigit(*w);

            if (m_word_stats.find(normalizedWord) == m_word_stats.end()) {
                m_word_stats.insert(std::pair<std::string, int>(normalizedWord, 1));
            } else {
                m_word_stats.at(normalizedWord) += 1;
            }
        }

        if ((numInstance + 1) % m_options.verboseIter == 0) {
            cout << numInstance + 1 << " ";
            if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
                cout << std::endl;
            cout.flush();
        }

        if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
            break;
    }
    std::cout << numInstance << " " << endl;

    return 0;
}

int Classifier::addTestAlpha(const vector<Instance> &vecInsts) {
    std::cout << "Adding word Alphabet..." << endl;
    int numInstance;
    for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
        const Instance *pInstance = &vecInsts.at(numInstance);

        vector<const string *> words;

        for (const string &w : pInstance->m_title_words) {
            words.push_back(&w);
        }

        for (const string *w : words) {
            string normalizedWord = normalize_to_lowerwithdigit(*w);

            if (m_word_stats.find(normalizedWord) == m_word_stats.end()) {
                m_word_stats.insert(std::pair<std::string, int>(normalizedWord, 0));
            } else {
                m_word_stats.at(normalizedWord) += 1;
            }
        }

        if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
            break;
    }
    cout << numInstance << " " << endl;

    return 0;
}


void Classifier::convert2Example(const Instance *pInstance, Example &exam) {
    exam.m_category = pInstance->m_category;
    Feature feature = Feature::valueOf(*pInstance);
    exam.m_feature = feature;
}

void Classifier::initialExamples(const vector<Instance> &vecInsts,
        vector<Example> &vecExams) {
    int numInstance;
    for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
        const Instance *pInstance = &vecInsts.at(numInstance);
        Example curExam;
        convert2Example(pInstance, curExam);
        vecExams.push_back(curExam);
    }
}

void Classifier::train(const string &trainFile, const string &devFile,
        const string &testFile, const string &modelFile,
        const string &optionFile) {
    if (optionFile != "")
        m_options.load(optionFile);
    m_options.showOptions();

    vector<Instance> rawtrainInsts = readInstancesFromFile(trainFile, devFile, testFile);
    vector<Instance> trainInsts;
    for (Instance &ins : rawtrainInsts) {
        trainInsts.push_back(ins);
    }

    createAlphabet(trainInsts);
    //if (!m_options.wordEmbFineTune) {
    //    addTestAlpha(devInsts);
    //    addTestAlpha(testInsts);
    //}

    bool bCurIterBetter = false;

    vector<Example> trainExamples, devExamples, testExamples;

    initialExamples(trainInsts, trainExamples);
    //initialExamples(devInsts, devExamples);
    //initialExamples(testInsts, testExamples);

    m_word_stats[unknownkey] = m_options.wordCutOff + 1;
    m_driver._modelparams.wordAlpha.initial(m_word_stats, m_options.wordCutOff);

    if (m_options.wordFile != "") {
        m_driver._modelparams.words.initial(&m_driver._modelparams.wordAlpha,
                m_options.wordFile, m_options.wordEmbFineTune);
    } else {
        m_driver._modelparams.words.initial(&m_driver._modelparams.wordAlpha,
                m_options.wordEmbSize, m_options.wordEmbFineTune);
    }

    m_driver._hyperparams.setRequared(m_options);
    m_driver.initial();

    dtype bestDIS = 0;

    srand(0);

    static vector<Example> subExamples;
    int devNum = devExamples.size(), testNum = testExamples.size();
    int non_exceeds_time = 0;
    auto time_start = std::chrono::high_resolution_clock::now();

    n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
    profiler.SetEnabled(true);
    profiler.BeginEvent("classifier train");

    for (int iter = 0; iter < 1; ++iter) {
        std::cout << "##### Iteration " << iter << std::endl;
        std::vector<int> indexes;
        //for (int i = 0; i < trainExamples.size(); ++i) {
        //    indexes.push_back(i);
        //}
        //std::random_shuffle(indexes.begin(), indexes.end());
        int batchBlock = trainExamples.size() / m_options.batchSize;
        batchBlock /= 10;
        if (trainExamples.size() % m_options.batchSize != 0)
            batchBlock++;
        Metric metric;
        for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
            subExamples.clear();
            int start_pos = updateIter * m_options.batchSize;
            int end_pos = (updateIter + 1) * m_options.batchSize;
            if (end_pos > trainExamples.size())
                end_pos = trainExamples.size();

            for (int idy = start_pos; idy < end_pos; idy++) {
                subExamples.push_back(trainExamples.at(idy));
            }

            int curUpdateIter = iter * batchBlock + updateIter;
            dtype cost = m_driver.train(subExamples, curUpdateIter);

            metric.overall_label_count += m_driver._metric.overall_label_count;
            metric.correct_label_count += m_driver._metric.correct_label_count;

            //m_driver.checkgrad(subExamples, updateIter);

            m_driver.updateModel();

            if (updateIter % 100 == 0) {
            std::cout << "current: " << updateIter + 1 << ", total block: "
                    << batchBlock << std::endl;
            metric.print();
            }
        }

        auto time_end = std::chrono::high_resolution_clock::now();
        std::cout << "Train finished. Total time taken is: "
            << std::chrono::duration<double>(time_end - time_start).count()
            << "s" << std::endl;
        //float accuracy = metric.getAccuracy();
        //std::cout << "train set acc:" << metric.getAccuracy() << std::endl;
    }
    profiler.EndEvent();
}

Category Classifier::predict(const Feature &feature, int excluded_class) {
    Category category;
    m_driver.predict(feature, category, excluded_class);
    return category;
}

//int main(int argc, char *argv[]) {
//    vector<Instance> instances = readInstancesFromFile("/home/wqs/news-title-classification/data/train_cla_utf8.txt");
//    for (Instance &ins : instances) {
//        std::cout << ins.m_category << std::endl;
//        for (string &w : ins.m_title_words) {
//            std::cout << w << "|";
//        }
//        std::cout << std::endl;
//    }

//    return 0;
//}

int main(int argc, char *argv[]) {
#if USE_GPU
    n3ldg_cuda::InitCuda();
#endif
    std::string trainFile = "", devFile = "", testFile = "", modelFile = "", optionFile = "";
    std::string outputFile = "";
    bool bTrain = false;
    int memsize = 0;
    dsr::Argument_helper ah;

    ah.new_flag("l", "learn", "train or test", bTrain);
    ah.new_named_string("train", "trainCorpus", "named_string",
            "training corpus to train a model, must when training", trainFile);
    ah.new_named_string("dev", "devCorpus", "named_string",
            "development corpus to train a model, optional when training", devFile);
    ah.new_named_string("test", "testCorpus", "named_string",
            "testing corpus to train a model or input file to test a model, optional when training and must when testing",
            testFile);
    ah.new_named_string("model", "modelFile", "named_string",
            "model file, must when training and testing", modelFile);
    ah.new_named_string("option", "optionFile", "named_string",
            "option file to train a model, optional when training", optionFile);
    ah.new_named_string("output", "outputFile", "named_string",
            "output file to test, must when testing", outputFile);
    ah.new_named_int("memsize", "memorySize", "named_int",
            "This argument decides the size of static memory allocation", memsize);

    ah.process(argc, argv);

    if (memsize < 0)
        memsize = 0;
    Classifier the_classifier(memsize);
    if (bTrain) {
        the_classifier.train(trainFile, devFile, testFile, modelFile, optionFile);
    } else {
    }
    n3ldg_cuda::Profiler::Ins().Print();
#if USE_GPU
    n3ldg_cuda::EndCuda();
#else
#endif
}
