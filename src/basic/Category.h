#ifndef NEWS_SRC_BASIC_CATEGORY_H
#define NEWS_SRC_BASIC_CATEGORY_H

#include <string>
#include <vector>
#include "N3LDG.h"

enum Category {
    VERY_NEGATIVE = 0,
    NEGATIVE = 1,
    NEUTRAL = 2,
    POSITIVE = 3,
    VERY_POSITIVE = 4
};

std::vector<dtype> ToVector(Category category) {
    std::vector<dtype> r;
    r.resize(5);
    for (int i=0; i<5; ++i) {
        r.at(i) = i == static_cast<int>(category);
    }
    return r;
}

#endif
