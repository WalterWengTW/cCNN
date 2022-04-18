#include <cmath>
#include <iostream>
#include "Layers.h"


/*
====================================================================================================================
=                                               Convolution Layer                                                  =
====================================================================================================================
*/

CONV2D::CONV2D(int Units, int KH, int KW, int SH, int SW, int Pad, int InputShape[3])
{
	Kernels = Units;
	KernelH = KH;
	KernelW = KW;
	StrideH = SH;
	StrideW = SW;
	Padding = Pad;
	InputChannel = InputShape[0];
	Kernel.Restructor(1, Kernels * InputChannel, KernelH, KernelW);
	Kernel.Randoms(1.0);
	Bias.Restructor(1, Kernels, 1, 1);
	Bias.Randoms(1.0);
	OutputShape[0] = Kernels;
	OutputShape[1] = (InputShape[1] + 2 * Padding - KernelH) / StrideH + 1;
	OutputShape[2] = (InputShape[2] + 2 * Padding - KernelW) / StrideW + 1;
}

void CONV2D::ImagePadding(Matrix* InputImage, int Pad, Matrix* OutputImage)
{
	OutputImage->Restructor(InputImage->Number, InputImage->Channel, InputImage->Height + 2 * Pad, InputImage->Width + 2 * Pad);
	for (int n = 0; n < OutputImage->Number; n++)
	{
		for (int c = 0; c < OutputImage->Channel; c++)
		{
			for (int h = 0; h < OutputImage->Height; h++)
			{
				for (int w = 0; w < OutputImage->Width; w++)
				{
					if ((h < Pad) || (h > OutputImage->Height - Pad - 1) || (w < Pad) || (w > OutputImage->Width - Pad - 1))
					{
						OutputImage->Array[n][c][h][w] = 0;
					}
					else
					{
						OutputImage->Array[n][c][h][w] = InputImage->Array[n][c][h - Pad][w - Pad];
					}
				}
			}
		}
	}
	
}

void CONV2D::Inference(Matrix* Image, Matrix* Output)
{
	double sum = 0;
	Matrix ImagePadded;
	ImagePadding(Image, Padding, &ImagePadded);
	Output->Restructor(Image->Number, OutputShape[0], OutputShape[1], OutputShape[2]);

	for (int n = 0; n < ImagePadded.Number; n++)
	{
		for (int k = 0; k < Kernels; k++)
		{
			for (int c = 0; c < ImagePadded.Channel; c++)
			{
				for (int hi = 0, ho = 0; (hi < ImagePadded.Height) && (ho < OutputShape[1]); hi = hi + StrideH, ho++)
				{
					if ((hi + KernelH) > ImagePadded.Height) break;
					for (int wi = 0, wo = 0; (wi < ImagePadded.Width) && (wo < OutputShape[2]); wi = wi +StrideW, wo++)
					{
						if ((wi + KernelW) > ImagePadded.Width) break;
						sum = 0;
						for (int kh = 0; kh < KernelH; kh++)
						{
							for (int kw = 0; kw < KernelW; kw++)
							{
								sum = sum + ImagePadded.Array[n][c][hi + kh][wi + kw] * Kernel.Array[0][k * ImagePadded.Channel + c][kh][kw] ;
							}
						}
						Output->Array[n][k][ho][wo] = Output->Array[n][k][ho][wo] + sum;
					}
				}
			}
		}
	}
	Matrix temp;
	temp.Add(Output, &Bias);
	Output->Copy(&temp);

	temp.ReleasMemory();
	ImagePadded.ReleasMemory();
}

void CONV2D::Learning(Matrix* dOut, Matrix* X, Matrix* dX)
{
	/* Bias gradient */
	//Matrix dBias;
	Matrix temp;
	Matrix temp1;
	temp.SumBy(dOut, 3);
	temp1.SumBy(&temp, 2);
	dBias.SumBy(&temp1, 0);


	/* Kernel gradient */
	//Matrix dKernel;
	Matrix X_Padded;
	ImagePadding(X, Padding, &X_Padded);

	dKernel.Copy(&Kernel);
	dKernel.Zeros();
	float sum = 0;
	for (int n = 0; n < X->Number; n++)
	{
		for (int k = 0; k < Kernels; k++)
		{
			for (int c = 0; c < X->Channel; c++)
			{
				for (int hi = 0, ho = 0; (hi < X_Padded.Height) && (ho < KernelH); hi = hi + StrideH, ho++) 
				{
					if ((hi + dOut->Height) > X_Padded.Height) break;
					for (int wi = 0, wo = 0; (wi < X_Padded.Width) && (wo < KernelW); wi = wi + StrideW, wo++)
					{
						if ((wi + dOut->Width) > X_Padded.Width) break;
						sum = 0;
						for (int douth = 0; douth < dOut->Height; douth++)
						{
							for (int doutw = 0; doutw < dOut->Width; doutw++)
							{
								sum = sum + X_Padded.Array[n][c][hi + douth][wi + doutw] * dOut->Array[n][k][douth][doutw];
							}
						}
						dKernel.Array[0][k * X->Channel + c][ho][wo] = dKernel.Array[0][k * X->Channel + c][ho][wo] + sum;
					}
				}
			}
		}
	}

	/* dOut turn into dX and backward propagating */
	Matrix dOut_Padded;
	ImagePadding(dOut, KernelH - 1, &dOut_Padded);
	dX->Restructor(X->Number, X->Channel, X->Height, X->Width);

	for (int n = 0; n < dOut->Number; n++)
	{
		for (int c = 0; c < X->Channel; c++)
		{
			for (int k = 0; k < Kernels; k++)
			{
				for (int hi = 0, ho = 0; (hi < dOut_Padded.Height) && (ho < X->Height + 2 * Padding); hi = hi + StrideH, ho++)
				{
					if (hi + KernelH > dOut_Padded.Height) break;
					for (int wi = 0, wo = 0; (wi < dOut_Padded.Width) && (wo < X->Width + 2 * Padding); wi = wi + StrideW, wo++)
					{
						if (wi + KernelW > dOut_Padded.Width) break;
						sum = 0;
						for (int kh = 0; kh < KernelH; kh++)
						{
							for (int kw = 0; kw < KernelW; kw++)
							{
								sum = sum + dOut_Padded.Array[n][k][hi + kh][wi + kw] * Kernel.Array[0][k * X->Channel + c][KernelH - 1 - kh][KernelW - 1 - kw];
							}
						}
						if ((ho >= Padding) && (ho <= X->Height + Padding - 1) && (wo >= Padding) && (wo <= X->Width + Padding -1))
							dX->Array[n][c][ho - Padding][wo - Padding] = dX->Array[n][c][ho - Padding][wo - Padding] + sum;
					}
				}
			}
		}
	}

	/* Release Memory */
	temp.ReleasMemory();
	temp1.ReleasMemory();
	X_Padded.ReleasMemory();
	dOut_Padded.ReleasMemory();
}


/*
====================================================================================================================
=                                                 Pooling Layer                                                    =
====================================================================================================================
*/

POOL2D::POOL2D(int PH, int PW, int SH, int SW, char PBy, int InShape[3])
{
	PoolH = PH;
	PoolW = PW;
	StrideH = SH;
	StrideW = SW;
	PoolBy = PBy;
	OutputShape[0] = InShape[0];
	OutputShape[1] = (InShape[1] - PoolH) / StrideH + 1;
	OutputShape[2] = (InShape[2] - PoolW) / StrideW + 1;
	InputShape[0] = InShape[0];
	InputShape[1] = InShape[1];
	InputShape[2] = InShape[2];
}

void POOL2D::Inference(Matrix* Image, Matrix* Output)
{
	Mask.Restructor(Image->Number, Image->Channel, Image->Height, Image->Width);
	Mask.Zeros();
	Output->Restructor(Image->Number, Image->Channel, OutputShape[1], OutputShape[2]);

	for (int n = 0; n < Image->Number; n++)
	{
		for (int c = 0; c < Image->Channel; c++)
		{
			for (int hi = 0, ho = 0; (hi < Image->Height) || (ho < OutputShape[1]); hi = hi + StrideH, ho++)
			{
				if ((hi + PoolH) > Image->Height) break;
				for (int wi = 0, wo = 0; (wi < Image->Width) || (wo < OutputShape[2]); wi = wi + StrideW, wo++)
				{
					if ((wi + PoolW) > Image->Width) break;
					if (PoolBy == 'M')
					{
						float MaxValue = -9999;
						int PositionH, PositionW;
						for (int ph = 0; ph < PoolH; ph++)
						{
							for (int pw = 0; pw < PoolW; pw++)
							{
								if (Image->Array[n][c][hi + ph][wi + pw] > MaxValue)
								{
									MaxValue = Image->Array[n][c][hi + ph][wi + pw];
									PositionH = ph;
									PositionW = pw;
								}
							}
						}
						Output->Array[n][c][ho][wo] = MaxValue;
						Mask.Array[n][c][hi + PositionH][wi + PositionW] = 1.0;
					}
					else if (PoolBy == 'm')
					{
						float MinValue = 9999;
						int PositionH, PositionW;
						for (int ph = 0; ph < PoolH; ph++)
						{
							for (int pw = 0; pw < PoolW; pw++)
							{
								if (Image->Array[n][c][hi + ph][wi + pw] < MinValue)
								{
									MinValue = Image->Array[n][c][hi + ph][wi + pw];
									PositionH = ph;
									PositionW = pw;
								}
							}
						}
						Output->Array[n][c][ho][wo] = MinValue;
						Mask.Array[n][c][hi + PositionH][wi + PositionW] = 1.0;
					}
					else if (PoolBy == 'A' || PoolBy == 'a')
					{
						float sum = 0;
						for (int ph = 0; ph < PoolH; ph++)
						{
							for (int pw = 0; pw < PoolW; pw++)
							{
								sum = sum + Image->Array[n][c][hi + ph][wi + pw] / (PoolH * PoolW);
								Mask.Array[n][c][hi + ph][wi + pw] = 1.0 / (PoolH * PoolW);
							}
						}
						Output->Array[n][c][ho][wo] = sum;
					}
				}
			}
		}
	}
}

void POOL2D::Learning(Matrix* dOut, Matrix* dX)
{
	dX->Restructor(dOut->Number, dOut->Channel, InputShape[1], InputShape[2]);
	dX->Zeros();
	for (int n = 0; n < dOut->Number; n++)
	{
		for (int c = 0; c < dOut->Channel; c++)
		{
			for (int h = 0; h < dOut->Height; h++)
			{
				for (int w = 0; w < dOut->Width; w++)
				{
					for (int ph = 0; ph < PoolH; ph++)
					{
						for (int pw = 0; pw < PoolW; pw++)
						{
							dX->Array[n][c][h * StrideH + ph][w * StrideW + pw] = dX->Array[n][c][h * StrideH + ph][w * StrideW + pw] +
								Mask.Array[n][c][h * StrideH + ph][w * StrideW + pw] * dOut->Array[n][c][h][w];
						}
					}
				}
			}
		}
	}
}

/*
====================================================================================================================
=                                              Activation Layer                                                    =
====================================================================================================================
*/

/*
======================================================
=                     ReLU                           =
======================================================
*/
RELU::RELU(int InputShape[3])
{
	OutputShape[0] = InputShape[0];
	OutputShape[1] = InputShape[1];
	OutputShape[2] = InputShape[2];
}
void RELU::Inference(Matrix* Input, Matrix* Output)
{
	Mask.Restructor(Input->Number, Input->Channel, Input->Height, Input->Width);
	Output->Restructor(Input->Number, Input->Channel, Input->Height, Input->Width);

	for (int n = 0; n < Input->Number; n++)
	{
		for (int c = 0; c < Input->Channel; c++)
		{
			for (int h = 0; h < Input->Height; h++)
			{
				for (int w = 0; w < Input->Width; w++)
				{
					if (Input->Array[n][c][h][w] > 0)
					{
						Output->Array[n][c][h][w] = Input->Array[n][c][h][w];
						Mask.Array[n][c][h][w] = 1.0;
					}
					else
					{
						Output->Array[n][c][h][w] = 0.0;
						Mask.Array[n][c][h][w] = 0.0;
					}
				}
			}
		}
	}
}

void RELU::Learning(Matrix* dOut, Matrix* dX)
{
	dX->Restructor(dOut->Number, dOut->Channel, dOut->Height, dOut->Width);

	for (int n = 0; n < dOut->Number; n++)
	{
		for (int c = 0; c < dOut->Channel; c++)
		{
			for (int h = 0; h < dOut->Height; h++)
			{
				for (int w = 0; w < dOut->Width; w++)
				{
					dX->Array[n][c][h][w] = dOut->Array[n][c][h][w] * Mask.Array[n][c][h][w];
				}
			}
		}
	}
}

/*
======================================================
=                     Softma                         =
======================================================
*/
SOFTMAX::SOFTMAX(int InShape[3])
{
	OutputShape[0] = InShape[0];
	OutputShape[1] = InShape[1];
	OutputShape[2] = InShape[2];
}

void SOFTMAX::Inference(Matrix* Input, Matrix* Output)
{
	Matrix temp;
	Matrix temp_max;
	Matrix temp_sum;

	temp.Copy(Input);
	temp_max.MaxBy(&temp, 3);
	temp.Sub(Input, &temp_max);

	for (int n = 0; n < Input->Number; n++)
	{
		for (int c = 0; c < Input->Channel; c++)
		{
			for (int h = 0; h < Input->Height; h++)
			{
				for (int w = 0; w < Input->Width; w++)
				{
					temp.Array[n][c][h][w] = exp(temp.Array[n][c][h][w]);
				}
			}
		}
	}
	temp_sum.SumBy(&temp, 3);
	temp_max.Div(&temp, &temp_sum);
	Output->Copy(&temp_max);

	temp.ReleasMemory();
	temp_max.ReleasMemory();
	temp_sum.ReleasMemory();
}

void SOFTMAX::LearningWithLoss(Matrix* Y_truth, Matrix* Y_pred, Matrix* dX)
{
	int Batch = Y_pred->Height;
	Matrix temp;

	if (Y_pred->Width == Y_truth->Width)
	{
		temp.Sub(Y_truth, Y_pred);
	}
	else
	{
		temp.Copy(Y_pred);
		for (int n = 0; n < Y_pred->Number; n++)
		{
			for (int c = 0; c < Y_pred->Channel; c++)
			{
				for (int h = 0; h < Y_pred->Height; h++)
				{
					for (int w = 0; w < Y_pred->Width; w++)
					{
						if (w == (int)Y_truth->Array[n][c][h][0])
							temp.Array[n][c][h][w] = temp.Array[n][c][h][w] - 1;
					}
				}
			}
		}
	}
	dX->Mul(&temp, 1.0 / (float)Batch);

	temp.ReleasMemory();
}

/*
====================================================================================================================
=                                                Flatten Layer                                                     =
====================================================================================================================
*/
FLATTEN::FLATTEN(int InShape[3])
{
	OutputShape[0] = 1;
	OutputShape[1] = 1;
	OutputShape[2] = InShape[0] * InShape[1] * InShape[2];
	InputShape[0] = InShape[0];
	InputShape[1] = InShape[1];
	InputShape[2] = InShape[2];

}
void FLATTEN::Inference(Matrix* Input, Matrix* Output)
{
	Batch = Input->Number;
	Output->Restructor(1, 1, Batch, Input->Channel * Input->Height * Input->Width);

	for (int n = 0; n < Batch; n++)
	{
		for (int c = 0; c < Input->Channel; c++)
		{
			for (int h = 0; h < Input->Height; h++)
			{
				for (int w = 0; w < Input->Width; w++)
				{
					Output->Array[0][0][n][c * Input->Height * Input->Width + h * Input->Width + w] = Input->Array[n][c][h][w];
				}
			}
		}
	}
}

void FLATTEN::Learning(Matrix* dOut, Matrix* dX)
{
	dX->Restructor(Batch, InputShape[0], InputShape[1], InputShape[2]);

	for (int n = 0; n < Batch; n++)
	{
		for (int c = 0; c < InputShape[0]; c++)
		{
			for (int h = 0; h < InputShape[1]; h++)
			{
				for (int w = 0; w < InputShape[2]; w++)
				{
					dX->Array[n][c][h][w] = dOut->Array[0][0][n][c * InputShape[1] * InputShape[2] + h * InputShape[2] + w];
				}
			}
		}
	}
}

/*
====================================================================================================================
=                                         Full Connection Layer                                                    =
====================================================================================================================
*/
DENSE::DENSE(int Units,int InShape[3])
{
	OutputShape[0] = InShape[0];
	OutputShape[1] = InShape[1];
	OutputShape[2] = Units;
	Neurons = Units;

	Weight.Restructor(1, 1, InShape[2], Neurons);
	Weight.Randoms(1);
	Bias.Restructor(1, 1, 1, Neurons);
	Bias.Randoms(1);
}

void DENSE::Inference(Matrix* Input, Matrix* Output)
{
	Output->Restructor(1, 1, Input->Number, Neurons);
	Matrix temp;
	temp.Dot2D(Input, &Weight);
	Output->Add(&temp, &Bias);
	temp.ReleasMemory();
}

void DENSE::Learning(Matrix* dOut, Matrix* X, Matrix* dX)
{
	dBias.SumBy(dOut, 2);
	Matrix temp;
	temp.Tsp2D(X);
	dWeight.Dot2D(&temp, dOut);
	temp.Tsp2D(&Weight);
	dX->Dot2D(dOut, &temp);

	temp.ReleasMemory();
}

/*
====================================================================================================================
=                                         Loss Function Layer                                                      =
====================================================================================================================
*/

void LOSSFUNCTION::OneHotEncoder(Matrix* Y_label, int classes, Matrix* Y_label_onehot)
{
	Matrix temp(Y_label->Number, Y_label->Channel, Y_label->Height, classes);
	
	for (int n = 0; n < Y_label->Number; n++)
	{
		for (int c = 0; c < Y_label->Channel; c++)
		{
			for (int h = 0; h < Y_label->Height; h++)
			{
				for (int cls = 0; cls < classes; cls++)
				{
					if (cls == (int)Y_label->Array[n][c][h][0])
						temp.Array[n][c][h][cls] = 1.0;
					else
						temp.Array[n][c][h][cls] = 0.0;
				}
			}
		}
	}
	Y_label_onehot->Copy(&temp);
	temp.ReleasMemory();
}

float LOSSFUNCTION::CrossEntropy(Matrix* Y_pred, Matrix* Y_label)
{
	Matrix temp;
	if (Y_pred->Width != Y_label->Width)
	{
		OneHotEncoder(Y_label, Y_pred->Width, &temp);
		float sum = 0;
		for (int n = 0; n < Y_pred->Number; n++)
		{
			for (int c = 0; c < Y_pred->Channel; c++)
			{
				for (int h = 0; h < Y_pred->Height; h++)
				{
					for (int w = 0; w < Y_pred->Width; w++)
					{
						sum = sum + temp.Array[n][c][h][w] * log(Y_pred->Array[n][c][h][w] + 0.0000001);
					}
				}
			}
		}
		temp.ReleasMemory();
		return -sum / Y_pred->Height;
	}
	else
	{
		float sum = 0;
		for (int n = 0; n < Y_pred->Number; n++)
		{
			for (int c = 0; c < Y_pred->Channel; c++)
			{
				for (int h = 0; h < Y_pred->Height; h++)
				{
					for (int w = 0; w < Y_pred->Width; w++)
					{
						sum = sum + Y_label->Array[n][c][h][w] * log(Y_pred->Array[n][c][h][w] + 0.0000001);
					}
				}
			}
		}
		return -sum / Y_pred->Height;
	}
}

/*
====================================================================================================================
=                                                     Adam                                                         =
====================================================================================================================
*/

ADAM::ADAM(float lr, float beta1, float beta2)
{
	LearningRate = lr;
	Beta1 = beta1;
	Beta2 = beta2;
	Iterator = 0;
}
void ADAM::Updator(Matrix* W, Matrix* dW, Matrix* B, Matrix* dB)
{
	if (Iterator == 0)
	{
		Mw.Copy(dW);
		Mw.Zeros();
		Vw.Copy(dW);
		Vw.Zeros();

		Mb.Copy(dB);
		Mb.Zeros();
		Vb.Copy(dB);
		Vb.Zeros();
	}
	Iterator++;
	float Lr_t = LearningRate * sqrt(1 - pow(Beta2, Iterator)) / (1 - pow(Beta1, Iterator));

	Matrix Mtemp, Mtemp1;
	Matrix Vtemp, Vtemp1, Vtemp2;
	Matrix temp1, temp2;

	Mtemp.Mul(&Mw, Beta1);
	Mtemp1.Mul(dW, (1 - Beta1));
	Mw.Add(&Mtemp, &Mtemp1); //Mt = Beta1 * Mt-1 + ( 1 -Beta1) * Gt

	Vtemp.Mul(&Vw, Beta2);
	Vtemp1.Mul(dW, dW);
	Vtemp2.Mul(&Vtemp1, (1 - Beta2));
	Vw.Add(&Vtemp, &Vtemp2); //Vt = Beta2 * Vt-1 + (1 - Beta2) * Gt^2

	Mtemp.Mul(&Mw, 1 / (1 - Beta1));
	Vtemp.Mul(&Vw, 1 / (1 - Beta2));
	
	temp1.Root(&Vtemp);
	temp2.Add(&temp1, 0.00000001);
	temp1.Reciprocal(&temp2);
	temp2.Mul(&temp1, Lr_t);
	temp1.Mul(&temp2, &Mtemp);

	temp2.Sub(W, &temp1);
	W->Copy(&temp2);

	//temp1.Mul(dW, LearningRate);
	//temp2.Sub(W, &temp1);
	//W->Copy(&temp2);
	
//======================================

	Mtemp.Mul(&Mb, Beta1);
	Mtemp1.Mul(dB, (1 - Beta1));
	Mb.Add(&Mtemp, &Mtemp1);

	Vtemp.Mul(&Vb, Beta2);
	Vtemp1.Mul(dB, dB);
	Vtemp2.Mul(&Vtemp1, (1 - Beta2));
	Vb.Add(&Vtemp, &Vtemp2);

	Mtemp.Mul(&Mb, 1 / (1 - Beta1));
	Vtemp.Mul(&Vb, 1 / (1 - Beta2));

	temp1.Root(&Vtemp);
	temp2.Add(&temp1, 0.00000001);
	temp1.Reciprocal(&temp2);
	temp2.Mul(&temp1, Lr_t);
	temp1.Mul(&temp2, &Mtemp);

	temp2.Sub(B, &temp1);
	B->Copy(&temp2);

	//temp1.Mul(dB, LearningRate);
	//temp2.Sub(B, &temp1);
	//B->Copy(&temp2);

	Mtemp.ReleasMemory();
	Mtemp1.ReleasMemory();
	Vtemp.ReleasMemory();
	Vtemp1.ReleasMemory();
	Vtemp2.ReleasMemory();
	temp1.ReleasMemory();
	temp2.ReleasMemory();
}