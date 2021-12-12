class Java {
private:
    void *data = NULL;
public:
    typedef struct {
        int width;
        int height;
    } Dims;
    bool init();
    void copyImage(float *buffer);
    Dims getImageDims();
};
