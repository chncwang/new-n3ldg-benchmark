#ifndef _JST_READER_
#define _JST_READER_

#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <regex>

#include "Instance.h"
#include "Targets.h"

std::vector<std::string> readLines(const std::string &fullFileName) {
    std::vector<string> lines;
    std::ifstream input(fullFileName);
    for (std::string line; getline(input, line);) {
        lines.push_back(line);
    }
    return lines;
}

int gWordCount = 0;

void readWordsToInstance(const std::string &line, Instance *instance) {
    std::vector<std::string> words;
    split_bychars(line, words);
    gWordCount += words.size();
    instance->m_title_words = std::move(words);
}

void readLabelToInstance(const std::string &parentLine,
        const std::string &labelLine, Instance *instance) {
    std::vector<std::string> parentStrings, labelStrings;
    split_bychars(parentLine, parentStrings);
    split_bychars(labelLine, labelStrings);
    bool foundRoot = false;
    std::vector<int> parents;
    for (int i = 0; i < parentStrings.size(); ++i) {
        std::string &w = parentStrings.at(i);
        if (w == "0") {
            foundRoot = true;
            instance->m_category =
                static_cast<Category>(2 + std::stoi(labelStrings.at(i)));
            instance->m_root = i;
        }
        int n = std::stoi(w);
        parents.push_back(n - 1);
    }
    instance->m_parents = parents;
    if (!foundRoot) {
        abort();
    }
}

std::vector<Instance> readInstancesFromFile(const std::string &textFile,
        const std::string &parentFile, const std::string &labelFile) {
    std::vector<std::string> textLines = readLines(textFile);
    std::vector<std::string> parentLines = readLines(parentFile);
    std::vector<std::string> labelLines = readLines(labelFile);

    std::vector<Instance> instances;
    instances.reserve(textLines.size());
    for (int i = 0; i < textLines.size(); ++i) {
        std::string &textLine = textLines.at(i);
        std::string &parentLine = parentLines.at(i);
        std::string &labelLine = labelLines.at(i);
        Instance instance;
        readWordsToInstance(textLine, &instance);
        readLabelToInstance(parentLine, labelLine, &instance);
        instances.push_back(instance);
    }
    std::cout << "gWordCount:" << gWordCount << std::endl;
    return instances;
}

#endif
