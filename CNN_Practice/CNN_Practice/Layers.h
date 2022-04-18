#include <iostream>
#include "Matrix.h"

class CONV2D {

private:

public:
	int KernelH, KernelW, StrideH, StrideW;
	int Kernels;
	int Padding;
	int InputChannel;
	int OutputShape[3];
	Matrix Kernel, Bias;
	Matrix dKernel, dBias;
	CONV2D(int, int, int, int, int, int, int*);
	void Inference(Matrix*, Matrix*);
	void Learning(Matrix*, Matrix*, Matrix*);
	void ImagePadding(Matrix*, int, Matrix*);
};

class POOL2D {

private:

public:
	int PoolH, PoolW, StrideH, StrideW;
	char PoolBy;
	int InputShape[3];
	int OutputShape[3];
	Matrix Mask;
	POOL2D(int, int, int, int, char, int*);
	void Inference(Matrix*, Matrix*);
	void Learning(Matrix*, Matrix*);
};

class RELU {

private:

public:
	int OutputShape[3];
	Matrix Mask;
	RELU(int*);
	void Inference(Matrix*, Matrix*);
	void Learning(Matrix*, Matrix*);
};

class SOFTMAX {

private:

public:
	int OutputShape[3];
	SOFTMAX(int*);
	void Inference(Matrix*, Matrix*);
	void LearningWithLoss(Matrix*, Matrix*, Matrix*);
};

class FLATTEN {

private:

public:
	int OutputShape[3];
	int InputShape[3];
	int Batch;
	FLATTEN(int*);
	void Inference(Matrix*, Matrix*);
	void Learning(Matrix*, Matrix*);
};

class DENSE {

private:

public:
	int OutputShape[3];
	int Neurons;
	Matrix Weight, Bias;
	Matrix dWeight, dBias;
	DENSE(int, int*);
	void Inference(Matrix*, Matrix*);
	void Learning(Matrix*, Matrix*, Matrix*);
};

class LOSSFUNCTION {

private:

public:
	float CrossEntropy(Matrix*, Matrix*);
	void OneHotEncoder(Matrix*, int, Matrix*);

};

class ADAM {

private:

public:
	float LearningRate, Beta1, Beta2;
	float Iterator;
	Matrix Mw, Vw, Mb, Vb;
	ADAM(float, float, float);
	void Updator(Matrix*, Matrix*, Matrix*, Matrix*);
};