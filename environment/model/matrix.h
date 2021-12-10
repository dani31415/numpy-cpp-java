#include <vector>

template <class T>
class matrix : public std::vector<std::vector<T>> {
private:
    int _m;
    int _n;
public:
    matrix(size_t m, size_t n);
    matrix();
    int m() { return _m; }
    int n() { return _n; }
    void resize2(size_t m, size_t n);
};

template<class T>
matrix<T>::matrix(size_t m, size_t n) {
    this->resize2(m,n);
}

template<class T>
matrix<T>::matrix() {
    this->resize2(0,0);
}

template<class T>
void matrix<T>::resize2(size_t m, size_t n) {
    this->_m = m;
    this->_n = n;
    this->resize(m);
    for (size_t i=0; i<this->size(); i++) {
        (*this)[i].resize(n);
    }
}
