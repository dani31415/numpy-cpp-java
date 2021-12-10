#include "country_side.h"

matrix<float> & CountrySide::getImage() {
    this->_img.resize2(10,15);
    for (int i=0;i<10;i++) {
        this->_img[i][i] = 1.f;
    }
    this->_img[0][0] = 0.3f;
    return this->_img;
}
