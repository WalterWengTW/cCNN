#include "Matrix.h"
#include <iostream>
using namespace std;

/* Constructor */
Matrix::Matrix(int n, int c, int h, int w)
{
	Number = n;
	Channel = c;
	Height = h;
	Width = w;
	Shape[0] = Number;
	Shape[1] = Channel;
	Shape[2] = Height;
	Shape[3] = Width;

	Array = new float*** [Number];
	
	for (int n = 0; n < Number; n++)
	{
		Array[n] = new float** [Channel];
		for (int c = 0; c < Channel; c++)
		{
			Array[n][c] = new float* [Height];
			for (int h = 0; h < Height; h++)
			{
				Array[n][c][h] = new float[Width];
			}
		}
	}
	Zeros();
}

Matrix::Matrix()
{
	Number = 1;
	Channel = 1;
	Height = 1;
	Width = 1;
	Shape[0] = Number;
	Shape[1] = Channel;
	Shape[2] = Height;
	Shape[3] = Width;

	Array = new float*** [Number];

	for (int n = 0; n < Number; n++)
	{
		Array[n] = new float** [Channel];
		for (int c = 0; c < Channel; c++)
		{
			Array[n][c] = new float* [Height];
			for (int h = 0; h < Height; h++)
			{
				Array[n][c][h] = new float[Width];
			}
		}
	}
	Zeros();
}

/* Mamber Function */

/*
=================================================
=              Initialize with zero             =
=================================================
*/

void Matrix::Zeros()
{
	for (int n = 0; n < Number; n++)
	{		
		for (int c = 0; c < Channel; c++)
		{			
			for (int h = 0; h < Height; h++)
			{
				for (int w = 0; w < Width; w++)
				{
					Array[n][c][h][w] = 0.0;
				}
			}
		}
	}
}

/*
=================================================
=           Initialize with Random              =
=================================================
*/

void Matrix::Randoms(float rate)
{
	for (int n = 0; n < Number; n++)
	{
		for (int c = 0; c < Channel; c++)
		{
			for (int h = 0; h < Height; h++)
			{
				for (int w = 0; w < Width; w++)
				{
					Array[n][c][h][w] = (((float)(rand()%2000)) - 1000.0) / 10000.0 * rate;
				}
			}
		}
	}
}

/*
=================================================
=      Initialize with Random  Integer          =
=================================================
*/

void Matrix::RandomInt(int Max)
{
	for (int n = 0; n < Number; n++)
	{
		for (int c = 0; c < Channel; c++)
		{
			for (int h = 0; h < Height; h++)
			{
				for (int w = 0; w < Width; w++)
				{
					Array[n][c][h][w] = (float)(rand()%Max);
				}
			}
		}
	}
}


/*
=================================================
=            Initialize with Ones               =
=================================================
*/

void Matrix::Ones()
{
	for (int n = 0; n < Number; n++)
	{
		for (int c = 0; c < Channel; c++)
		{
			for (int h = 0; h < Height; h++)
			{
				for (int w = 0; w < Width; w++)
				{
					Array[n][c][h][w] = 1.0;
				}
			}
		}
	}
}



/*
=================================================
=                    Restructor                 =
=================================================
*/

void Matrix::Restructor(int nt, int ct, int ht, int wt)
{
	for (int n = 0; n < Number; n++)
	{
		for (int c = 0; c < Channel; c++)
		{
			for (int h = 0; h < Height; h++)
			{
				delete[] Array[n][c][h];
			}
			delete[] Array[n][c];
		}
		delete[] Array[n];
	}
	delete[] Array;

	Number = nt;
	Channel = ct;
	Height = ht;
	Width = wt;
	Shape[0] = Number;
	Shape[1] = Channel;
	Shape[2] = Height;
	Shape[3] = Width;

	Array = new float*** [Number];

	for (int n = 0; n < Number; n++)
	{
		Array[n] = new float** [Channel];
		for (int c = 0; c < Channel; c++)
		{
			Array[n][c] = new float* [Height];
			for (int h = 0; h < Height; h++)
			{
				Array[n][c][h] = new float[Width];
			}
		}
	}
	Zeros();
}

/*
=================================================
=                     Copy                      =
=================================================
*/
void Matrix::Copy(Matrix* A)
{
	Restructor(A->Number, A->Channel, A->Height, A->Width);
	for (int n = 0; n < A->Number; n++)
	{
		for (int c = 0; c < A->Channel; c++)
		{
			for (int h = 0; h < A->Height; h++)
			{
				for (int w = 0; w < A->Width; w++)
				{
					Array[n][c][h][w] = A->Array[n][c][h][w];
				}
			}
		}
	}
}

/*
=================================================
=                Release Memory                 =
=================================================
*/

void Matrix::ReleasMemory()
{
	for (int n = 0; n < Number; n++)
	{
		for (int c = 0; c < Channel; c++)
		{
			for (int h = 0; h < Height; h++)
			{
				delete[] Array[n][c][h];
			}
			delete[] Array[n][c];
		}
		delete[] Array[n];
	}
	delete[] Array;
	Number = 0;
	Channel = 0;
	Height = 0;
	Width = 0;
}

/*
=================================================
=                    Display                    =
=================================================
*/

void Matrix::show_shape()
{
	cout << "\n" << "(" << Shape[0] << ", " << Shape[1] << ", " << Shape[2] << ", " << Shape[3] << ")" << endl;
}

void Matrix::show_element()
{
	cout << endl;
	for (int n = 0; n < Number; n++)
	{
		cout << "[";
		for (int c = 0; c < Channel; c++)
		{
			cout << "[";
			for (int h = 0; h < Height; h++)
			{
				cout << "[";
				for (int w = 0; w < Width; w++)
				{
					if (w == (Width - 1)) cout << Array[n][c][h][w];
					else cout << Array[n][c][h][w] << ", ";
				}
				cout << "]";
				if (h != (Height - 1)) cout << "," << endl;
			}
			cout << "]";
			if (c != (Channel - 1)) cout << "," << endl;

		}
		cout << "]";
		if (n != (Number - 1)) cout << "," << endl;
	}
	cout << endl;
}

/*
=================================================
=                    Addition                   =
=================================================
*/

void Matrix::Add(Matrix* A, float x)
{
	Restructor(A->Number, A->Channel, A->Height, A->Width);
	for (int n = 0; n < Number; n++)
	{
		for (int c = 0; c < Channel; c++)
		{
			for (int h = 0; h < Height; h++)
			{
				for (int w = 0; w < Width; w++)
				{
					Array[n][c][h][w] = A->Array[n][c][h][w] + x;
				}
			}
		}
	}
}

void Matrix::Add(Matrix* A, Matrix* B)
{
	Restructor(A->Number, A->Channel, A->Height, A->Width);
	

	for (int n = 0; n < Number; n++)
	{
		for (int c = 0; c < Channel; c++)
		{
			for (int h = 0; h < Height; h++)
			{
				for (int w = 0; w < Width; w++)
				{
					Array[n][c][h][w] = A->Array[n][c][h][w] + B->Array[n % B->Number][c % B->Channel][h % B->Height][w % B->Width];
				}
			}
		}
	}
}

/*
=================================================
=                    Substrct                   =
=================================================
*/


void Matrix::Sub(Matrix* A, Matrix* B)
{
	Restructor(A->Number, A->Channel, A->Height, A->Width);

	for (int n = 0; n < Number; n++)
	{
		for (int c = 0; c < Channel; c++)
		{
			for (int h = 0; h < Height; h++)
			{
				for (int w = 0; w < Width; w++)
				{
					Array[n][c][h][w] = A->Array[n][c][h][w] - B->Array[n % B->Number][c % B->Channel][h % B->Height][w % B->Width];
				}
			}
		}
	}
}

/*
=================================================
=                    Multiply                   =
=================================================
*/

void Matrix::Mul(Matrix* A, float x)
{
	Restructor(A->Number, A->Channel, A->Height, A->Width);
	for (int n = 0; n < Number; n++)
	{
		for (int c = 0; c < Channel; c++)
		{
			for (int h = 0; h < Height; h++)
			{
				for (int w = 0; w < Width; w++)
				{
					Array[n][c][h][w] = A->Array[n][c][h][w] * x;
				}
			}
		}
	}
}

void Matrix::Mul(Matrix* A, Matrix* B)
{
	Restructor(A->Number, A->Channel, A->Height, A->Width);

	for (int n = 0; n < Number; n++)
	{
		for (int c = 0; c < Channel; c++)
		{
			for (int h = 0; h < Height; h++)
			{
				for (int w = 0; w < Width; w++)
				{
					Array[n][c][h][w] = A->Array[n][c][h][w] * B->Array[n % B->Number][c % B->Channel][h % B->Height][w % B->Width];
				}
			}
		}
	}
}

/*
=================================================
=                    Division                   =
=================================================
*/

void Matrix::Div(Matrix* A, float x)
{
	Restructor(A->Number, A->Channel, A->Height, A->Width);
	for (int n = 0; n < Number; n++)
	{
		for (int c = 0; c < Channel; c++)
		{
			for (int h = 0; h < Height; h++)
			{
				for (int w = 0; w < Width; w++)
				{
					Array[n][c][h][w] = A->Array[n][c][h][w] / x;
				}
			}
		}
	}
}

void Matrix::Div(Matrix* A, Matrix* B)
{
	Restructor(A->Number, A->Channel, A->Height, A->Width);

	for (int n = 0; n < Number; n++)
	{
		for (int c = 0; c < Channel; c++)
		{
			for (int h = 0; h < Height; h++)
			{
				for (int w = 0; w < Width; w++)
				{
					Array[n][c][h][w] = A->Array[n][c][h][w] /( 0.0000001 + B->Array[n % B->Number][c % B->Channel][h % B->Height][w % B->Width]);
				}
			}
		}
	}
}

/*
=================================================
=                      Dot                      =
=================================================
*/
void Matrix::Dot2D(Matrix* A, Matrix* B)
{
	int NumberMax = A->Number;
	int ChannelMax = A->Channel;
	if (B->Number > NumberMax) NumberMax = B->Number;
	if (B->Channel > ChannelMax) ChannelMax = B->Channel;

	Restructor(NumberMax, ChannelMax, A->Height, B->Width);
	// (n x m) dot (m x k) = (n x k)  
	float sum = 0;
	for (int n = 0; n < NumberMax; n++)
	{
		for (int c = 0; c < ChannelMax; c++)
		{
			for (int h = 0; h < Height; h++)
			{
				for (int wb = 0; wb < B->Width; wb++)
				{
					sum = 0;
					for (int w = 0; w < A->Width; w++)
					{
						sum = sum + A->Array[n % A->Number][c % A->Channel][h][w] * B->Array[n % B->Number][c % B->Channel][w][wb];
					}
					Array[n][c][h][wb] = sum;
				}
			}
		}
	}
}

/*
=================================================
=             Maximum Value by Axis             =
=================================================
*/
void Matrix::MaxBy(Matrix* A, int Axis)
{
	if (Axis == 3)
	{
		Restructor(A->Number, A->Channel, A->Height, 1);
		for (int n = 0; n < A->Number; n++)
		{
			for (int c = 0; c < A->Channel; c++)
			{
				for (int h = 0; h < A->Height; h++)
				{
					Array[n][c][h][0] = -99999;
					for (int w = 0; w < A->Width; w++)
					{
						if (A->Array[n][c][h][w] > Array[n][c][h][0])  Array[n][c][h][0] = A->Array[n][c][h][w];
						else Array[n][c][h][0] = Array[n][c][h][0];
					}
				}
			}
		}
	}
	else if (Axis == 2)
	{
		Restructor(A->Number, A->Channel, 1, A->Width);
		for (int n = 0; n < A->Number; n++)
		{
			for (int c = 0; c < A->Channel; c++)
			{
				for (int w = 0; w < A->Width; w++)
				{
					Array[n][c][0][w] = -99999;
					for (int h = 0; h < A->Height; h++)
					{
						if (A->Array[n][c][h][w] > Array[n][c][0][w])  Array[n][c][0][w] = A->Array[n][c][h][w];
						else Array[n][c][0][w] = Array[n][c][0][w];
					}
				}
			}
		}
	}
	else if (Axis == 1)
	{
		Restructor(A->Number, 1, A->Height, A->Width);
		for (int n = 0; n < A->Number; n++)
		{
			for (int h = 0; h < A->Height; h++)
			{
				for (int w = 0; w < A->Width; w++)
				{
					Array[n][0][h][w] = -99999;
					for (int c = 0; c < A->Channel; c++)
					{
						if (A->Array[n][c][h][w] > Array[n][0][h][w])  Array[n][0][h][w] = A->Array[n][c][h][w];
						else Array[n][0][h][w] = Array[n][0][h][w];
					}
				}
			}
		}
	}
	else
	{
		Restructor(1, A->Channel, A->Height, A->Width);
		for (int c = 0; c < A->Channel; c++)
		{
			for (int h = 0; h < A->Height; h++)
			{
				for (int w = 0; w < A->Width; w++)
				{
					Array[0][c][h][w] = -99999;
					for (int n = 0; n < A->Number; n++)
					{
						if (A->Array[n][c][h][w] > Array[0][c][h][w])  Array[0][c][h][w] = A->Array[n][c][h][w];
						else Array[0][c][h][w] = Array[0][c][h][w];
					}
				}
			}
		}
	}
}

/*
=================================================
=                Sum Value by Axis              =
=================================================
*/

void Matrix::SumBy(Matrix* A, int Axis)
{
	float sum;
	if (Axis == 3)
	{
		Restructor(A->Number, A->Channel, A->Height, 1);
		for (int n = 0; n < A->Number; n++)
		{
			for (int c = 0; c < A->Channel; c++)
			{
				for (int h = 0; h < A->Height; h++)
				{
					sum = 0;
					for (int w = 0; w < A->Width; w++)
					{
						sum = sum + A->Array[n][c][h][w];
					}
					Array[n][c][h][0] = sum;
				}
			}
		}
	}
	else if (Axis == 2)
	{
		Restructor(A->Number, A->Channel, 1, A->Width);
		for (int n = 0; n < A->Number; n++)
		{
			for (int c = 0; c < A->Channel; c++)
			{
				for (int w = 0; w < A->Width; w++)
				{
					sum = 0;
					for (int h = 0; h < A->Height; h++)
					{
						sum = sum + A->Array[n][c][h][w];
					}
					Array[n][c][0][w] = sum;
				}
			}
		}
	}
	else if (Axis == 1)
	{
		Restructor(A->Number, 1, A->Height, A->Width);
		for (int n = 0; n < A->Number; n++)
		{
			for (int h = 0; h < A->Height; h++)
			{
				for (int w = 0; w < A->Width; w++)
				{
					sum = 0;
					for (int c = 0; c < A->Channel; c++)
					{
						sum = sum + A->Array[n][c][h][w];
					}
					Array[n][0][h][w] = sum;
				}
			}
		}
	}
	else
	{
		Restructor(1, A->Channel, A->Height, A->Width);
		for (int c = 0; c < A->Channel; c++)
		{
			for (int h = 0; h < A->Height; h++)
			{
				for (int w = 0; w < A->Width; w++)
				{
					sum = 0;
					for (int n = 0; n < A->Number; n++)
					{
						sum = sum + A->Array[n][c][h][w];
					}
					Array[0][c][h][w] = sum;
				}
			}
		}
	}
}

/*
=================================================
=   Transpose by dialog of lower 2 dimensions   =
=================================================
*/

void Matrix::Tsp2D(Matrix* A)
{
	Restructor(A->Number, A->Channel, A->Width, A->Height);
	for (int n = 0; n < A->Number; n++)
	{
		for (int c = 0; c < A->Channel; c++)
		{
			for (int h = 0; h < A->Height; h++)
			{
				for (int w = 0; w < A->Width; w++)
				{
					Array[n][c][w][h] = A->Array[n][c][h][w];
				}
			}
		}
	}
}

/*
=================================================
=            Root element by element            =
=================================================
*/

void Matrix::Root(Matrix* A)
{
	Restructor(A->Number, A->Channel, A->Height, A->Width);
	for (int n = 0; n < A->Number; n++)
	{
		for (int c = 0; c < A->Channel; c++)
		{
			for (int h = 0; h < A->Height; h++)
			{
				for (int w = 0; w < A->Width; w++)
				{
					Array[n][c][h][w] = sqrt(A->Array[n][c][h][w]);
				}
			}
		}
	}
}

/*
=================================================
=         Reciprocal element by element         =
=================================================
*/
void Matrix::Reciprocal(Matrix* A)
{
	Restructor(A->Number, A->Channel, A->Height, A->Width);
	for (int n = 0; n < A->Number; n++)
	{
		for (int c = 0; c < A->Channel; c++)
		{
			for (int h = 0; h < A->Height; h++)
			{
				for (int w = 0; w < A->Width; w++)
				{
					Array[n][c][h][w] = 1 / (0.0000001 + A->Array[n][c][h][w]);
				}
			}
		}
	}
}

/*
=================================================
=                Slicing by axis                =
=================================================
*/
void Matrix::Slice(Matrix* A, int from, int to, int axis)
{
	int SliceSize = to - from;

	if (axis == 3)
	{
		Restructor(A->Number, A->Channel, A->Height, SliceSize);
		for (int n = 0; n < Number; n++)
		{
			for (int c = 0; c < Channel; c++)
			{
				for (int h = 0; h < Height; h++)
				{
					for (int s = 0; s < SliceSize; s++)
					{
						Array[n][c][h][s] = A->Array[n][c][h][s + from];
					}
				}
			}
		}
	}
	else if (axis == 2)
	{
		Restructor(A->Number, A->Channel, SliceSize, A->Width);
		for (int n = 0; n < Number; n++)
		{
			for (int c = 0; c < Channel; c++)
			{
				for (int s = 0; s < SliceSize; s++)
				{
					for (int w = 0; w < Width; w++)
					{
						Array[n][c][s][w] = A->Array[n][c][s + from][w];
					}
				}
			}
		}
	}
	else if (axis == 1)
	{
		Restructor(A->Number, SliceSize, A->Height, A->Width);
		for (int n = 0; n < Number; n++)
		{
			for (int s = 0; s < SliceSize; s++)
			{
				for (int h = 0; h < Height; h++)
				{
					for (int w = 0; w < Width; w++)
					{
						Array[n][s][h][w] = A->Array[n][s + from][h][w];
					}
				}
			}
		}
	}
	else
	{
		Restructor(SliceSize, A->Channel, A->Height, A->Width);
		for (int s = 0; s < SliceSize; s++)
		{
			for (int c = 0; c < Channel; c++)
			{
				for (int h = 0; h < Height; h++)
				{
					for (int w = 0; w < Width; w++)
					{
						Array[s][c][h][w] = A->Array[s + from][c][h][w];
					}
				}
			}
		}
	}
}

/*
=================================================
=             Argument of Maximum               =
=================================================
*/

void Matrix::Argmax(Matrix* A, int Axis)
{
	if (Axis == 3)
	{
		Restructor(A->Number, A->Channel, A->Height, 1);
		for (int n = 0; n < A->Number; n++)
		{
			for (int c = 0; c < A->Channel; c++)
			{
				for (int h = 0; h < A->Height; h++)
				{
					Array[n][c][h][0] = 0;
					float MaxValue = -99999;
					for (int w = 0; w < A->Width; w++)
					{
						if (A->Array[n][c][h][w] > MaxValue)
						{
							Array[n][c][h][0] = w;
							MaxValue = A->Array[n][c][h][w];
						}
					}
				}
			}
		}
	}
	else if (Axis == 2)
	{
		Restructor(A->Number, A->Channel, 1, A->Width);
		for (int n = 0; n < A->Number; n++)
		{
			for (int c = 0; c < A->Channel; c++)
			{
				for (int w = 0; w < A->Width; w++)
				{
					Array[n][c][0][w] = 0;
					float MaxValue = -99999;
					for (int h = 0; h < A->Height; h++)
					{
						if (A->Array[n][c][h][w] > MaxValue)
						{
							Array[n][c][0][w] = h;
							MaxValue = A->Array[n][c][h][w];
						}
					}
				}
			}
		}
	}
	else if (Axis == 1)
	{
		Restructor(A->Number, 1, A->Height, A->Width);
		for (int n = 0; n < A->Number; n++)
		{
			for (int h = 0; h < A->Height; h++)
			{
				for (int w = 0; w < A->Width; w++)
				{
					Array[n][0][h][w] = 0;
					float MaxValue = -99999;
					for (int c = 0; c < A->Channel; c++)
					{
						if (A->Array[n][c][h][w] > MaxValue)
						{
							Array[n][0][h][w] = c;
							MaxValue = A->Array[n][c][h][w];
						}
					}
				}
			}
		}
	}
	else
	{
		Restructor(1, A->Channel, A->Height, A->Width);
		for (int c = 0; c < A->Channel; c++)
		{
			for (int h = 0; h < A->Height; h++)
			{
				for (int w = 0; w < A->Width; w++)
				{
					Array[0][c][h][w] = 0;
					float MaxValue = -99999;
					for (int n = 0; n < A->Number; n++)
					{
						if (A->Array[n][c][h][w] > MaxValue)
						{
							Array[0][c][h][w] = n;
							MaxValue = A->Array[n][c][h][w];
						}
					}
				}
			}
		}
	}
	
}