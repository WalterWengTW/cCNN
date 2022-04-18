#include <iostream>
using namespace std;

class Matrix {


public:
	/* Variable */
	int Height, Width, Channel, Number;
	int Shape[4];
	float**** Array;

	/* Constructor */
	Matrix(int, int, int, int);
	Matrix();
	
	/* Member Function*/
	void Zeros();
	void Ones();
	void Randoms(float);
	void RandomInt(int);
	void ReleasMemory();
	void Restructor(int, int, int, int);
	void Copy(Matrix*);
	void show_shape();
	void show_element();

	void Add(Matrix*, float);
	void Add(Matrix*, Matrix*);

	void Sub(Matrix*, Matrix*);

	void Mul(Matrix*, float);
	void Mul(Matrix*, Matrix*);

	void Div(Matrix*, float);
	void Div(Matrix*, Matrix*);

	void Dot2D(Matrix*, Matrix*);

	void MaxBy(Matrix*, int);
	void SumBy(Matrix*, int);

	void Tsp2D(Matrix*);

	void Root(Matrix*);

	void Reciprocal(Matrix*);

	void Slice(Matrix*, int, int, int);

	void Argmax(Matrix*, int);

};
