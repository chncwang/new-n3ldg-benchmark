#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "Utf.h"
#include "MyLib.h"
#include "Concat.h"
#include "UniOP.h"
#include "BiOP.h"
#include "LSTM1.h"
#include "DEPLSTM1.h"
#include <array>

class GraphBuilder {
public:
    std::vector<LookupNode> _input_nodes;
    TreeLSTM1Builder _tree_lstm_builder;
    LinearNode _neural_output;

    Graph *_graph;
    ModelParams *_modelParams;
    const static int max_sentence_length = 100;

    GraphBuilder() = default;
    GraphBuilder(const GraphBuilder&) = default;
    GraphBuilder(GraphBuilder&&) = default;

    void createNodes(int length_upper_bound) {
        _input_nodes.resize(length_upper_bound);
        _tree_lstm_builder.resize(length_upper_bound);
    }

    void initial(Graph *pcg, ModelParams &model, HyperParams &opts) {
        _graph = pcg;
        for (LookupNode &n : _input_nodes) {
            n.init(opts.wordDim, opts.dropProb);
            n.setParam(&model.words);
        }

        _tree_lstm_builder.init(&model.tree_lstm, opts.dropProb, true);

        _neural_output.init(opts.labelSize, -1);
        _neural_output.setParam(&model.olayer_linear);
        _modelParams = &model;
    }

    inline void forward(const Feature &feature, bool bTrain = false) {
        _graph->train = bTrain;
        for (int i = 0; i < feature.m_title_words.size(); ++i) {
            const std::string &word = feature.m_title_words.at(i);
            _input_nodes.at(i).forward(_graph, word);
        }

        std::vector<Node*> input_node_ptrs =
            toPointers<LookupNode, Node>(_input_nodes,
                    feature.m_title_words.size());

        _tree_lstm_builder.forward(_graph, input_node_ptrs, feature.m_parents);

        _neural_output.forward(_graph, &_tree_lstm_builder._hiddens.at(feature.m_root));
    }
};


#endif /* SRC_ComputionGraph_H_ */
