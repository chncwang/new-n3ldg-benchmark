#ifndef _EXAMPLE_H_
#define _EXAMPLE_H_

#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include "Targets.h"
#include "Instance.h"
#include "Category.h"

using namespace std;

class Feature
{
public:
    vector<std::string> m_title_words;
    vector<int> m_parents;
    int m_root;

    static Feature valueOf(const Instance &ins) {
        Feature feature;
        feature.m_title_words = ins.m_title_words;
        feature.m_parents = ins.m_parents;
        feature.m_root = ins.m_root;
        return feature;
    }
};

class Example
{
public:
    Feature m_feature;
    Category m_category;
};

#endif /*_EXAMPLE_H_*/
