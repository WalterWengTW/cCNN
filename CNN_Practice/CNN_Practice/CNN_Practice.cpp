
#include "Layers.h"
#include <iostream>
#include <fstream>
#include <time.h>
using namespace std;

void loadMNIST(Matrix*, Matrix*, Matrix*, Matrix*);
struct image {
	unsigned char header[54];
	unsigned char data[28][28][3];
};

int main()
{
	srand(time(NULL));
	
	/*==================================================================================================================
	=                                         LeNet-5 Architecture                                                     =
	===================================================================================================================*/
	

	int ImageShape[3] = { 1,28,28 };
	CONV2D Conv1(6, 5, 5, 1, 1, 2, ImageShape);
	RELU   ReLU1(Conv1.OutputShape);
	
	POOL2D Pool1(2, 2, 2, 2, 'M', ReLU1.OutputShape);
	
	CONV2D Conv2(16, 5, 5, 1, 1, 0, Pool1.OutputShape);
	RELU   ReLU2(Conv2.OutputShape);
	
	POOL2D Pool2(2, 2, 2, 2, 'M', ReLU2.OutputShape);
	
	CONV2D Conv3(120, 5, 5, 1, 1, 0, Pool2.OutputShape);
	RELU   ReLU3(Conv3.OutputShape);

	FLATTEN Flatten(ReLU3.OutputShape);

	DENSE Dense(10, Flatten.OutputShape);

	SOFTMAX Softmax(Dense.OutputShape);

	LOSSFUNCTION Evaluator;

	ADAM OptConv1(0.01, 0.9, 0.999);
	ADAM OptConv2(0.01, 0.9, 0.999);
	ADAM OptConv3(0.01, 0.9, 0.999);
	ADAM OptDense(0.01, 0.9, 0.999);

	cout << "LeNet-5 Achitecture\n";
	cout << "Conv 1 : \nOutput = (" << Conv1.OutputShape[0] << ", " << Conv1.OutputShape[1] << ", " << Conv1.OutputShape[2] << ")\n";
	cout << "Kernel : " << Conv1.Kernels << " @(" << Conv1.KernelH << "x" << Conv1.KernelW << ")\n\n";
	cout << "Pool 1 : \nOutput = (" << Pool1.OutputShape[0] << ", " << Pool1.OutputShape[1] << ", " << Pool1.OutputShape[2] << ")\n\n";
	cout << "Conv 2 : \nOutput = (" << Conv2.OutputShape[0] << ", " << Conv2.OutputShape[1] << ", " << Conv2.OutputShape[2] << ")\n";
	cout << "Kernel : " << Conv2.Kernels << " @(" << Conv2.KernelH << "x" << Conv2.KernelW << ")\n\n";
	cout << "Pool 2 : \nOutput = (" << Pool2.OutputShape[0] << ", " << Pool2.OutputShape[1] << ", " << Pool2.OutputShape[2] << ")\n\n";
	cout << "Conv 3 : \nOutput = (" << Conv3.OutputShape[0] << ", " << Conv3.OutputShape[1] << ", " << Conv3.OutputShape[2] << ")\n";
	cout << "Kernel : " << Conv3.Kernels << " @(" << Conv3.KernelH << "x" << Conv3.KernelW << ")\n\n";
	cout << "Dense : \nOutput = (" << Dense.OutputShape[0] << ", " << Dense.OutputShape[1] << ", " << Dense.OutputShape[2] << ")\n\n";

	/*==================================================================================================================
	=                                         MNIST load in                                                            =
	===================================================================================================================*/
	
	Matrix TrainImage(60000, 1, 28, 28);
	Matrix TrainLabel(1, 1, 60000, 1);
	Matrix TestImage(10000, 1, 28, 28);
	Matrix TestLabel(1, 1, 10000, 1);
	loadMNIST(&TrainImage, &TrainLabel, &TestImage, &TestLabel);

	cout << "================================================================\n";
	for (int r = 0; r < 28; r++)
	{
		for (int c = 0; c < 28; c++)
		{
			if (TrainImage.Array[4501][0][r][c] > 0) cout << "FF";
			else cout << "  ";
		}
		cout << endl;
	}
	cout << "Label : " << TrainLabel.Array[0][0][4501][0] << endl;
	cout << "================================================================\n";

	/*==================================================================================================================
	=                                              Training			                                                   =
	===================================================================================================================*/

	/*  Setting */
	int Epochs = 1;
	int BatchSize = 64;
	int BatchNumber = TrainImage.Number / BatchSize;

	if (TrainImage.Number % BatchSize != 0) BatchNumber++;
	
	/* index and variable data*/
	int epoch = 0;
	int batch = 0;
	int Nrest = TrainImage.Number;
	float losses;
	Matrix TrainInput, TrainTarget;
	Matrix FeatureMapL1, FeatureMapL1ReLU, FeatureMapL1SP;
	Matrix FeatureMapL2, FeatureMapL2ReLU, FeatureMapL2SP;
	Matrix FeatureMapL3, FeatureMapL3ReLU;
	Matrix Flattened, Dense1, Possibility;

	Matrix dConv1, dReLU1, dPool1;
	Matrix dConv2, dReLU2, dPool2;
	Matrix dConv3, dReLU3;
	Matrix dFlatten, dDense, dSoftmax;

	
	/* Training */
	while (epoch < Epochs)
	{
		batch = 0;
		Nrest = TrainImage.Number;
		while (batch < BatchNumber)
		{
			/* Batch Data */
			if (Nrest >= BatchSize)
			{
				Matrix temp;
				temp.Slice(&TrainImage, batch * BatchSize, (batch + 1) * BatchSize, 0);
				TrainInput.Mul(&temp, (1.0 / 255.0));
				TrainTarget.Slice(&TrainLabel, batch * BatchSize, (batch + 1) * BatchSize, 2);
				temp.ReleasMemory();
			}
			else
			{
				Matrix temp;
				temp.Slice(&TrainImage, TrainImage.Number - Nrest, TrainImage.Number, 0);
				TrainInput.Mul(&temp, (1.0 / 255.0));
				TrainTarget.Slice(&TrainLabel, TrainImage.Number - Nrest, TrainImage.Number, 2);
				temp.ReleasMemory();
			}


			/* Inference */
			Conv1.Inference(&TrainInput, &FeatureMapL1);
			ReLU1.Inference(&FeatureMapL1, &FeatureMapL1ReLU);
			Pool1.Inference(&FeatureMapL1ReLU, &FeatureMapL1SP);
			Conv2.Inference(&FeatureMapL1SP, &FeatureMapL2);
			ReLU2.Inference(&FeatureMapL2, &FeatureMapL2ReLU);
			Pool2.Inference(&FeatureMapL2ReLU, &FeatureMapL2SP);
			Conv3.Inference(&FeatureMapL2SP, &FeatureMapL3);
			ReLU3.Inference(&FeatureMapL3, &FeatureMapL3ReLU);
			Flatten.Inference(&FeatureMapL3ReLU, &Flattened);
			Dense.Inference(&Flattened, &Dense1);
			Softmax.Inference(&Dense1, &Possibility);

			/* Evaluation */
			losses = Evaluator.CrossEntropy(&Possibility, &TrainTarget);
			cout << batch + 1 << "th batch : " << losses << endl;
			//Possibility.show_element();
			

			/* Back Propagation and Learning */
			Softmax.LearningWithLoss(&TrainTarget, &Possibility, &dSoftmax);

			Dense.Learning(&dSoftmax, &Flattened, &dDense);
			OptDense.Updator(&Dense.Weight, &Dense.dWeight, &Dense.Bias, &Dense.dBias);

			Flatten.Learning(&dDense, &dFlatten);

			ReLU3.Learning(&dFlatten, &dReLU3);
			Conv3.Learning(&dReLU3, &FeatureMapL2SP, &dConv3);
			OptConv3.Updator(&Conv3.Kernel, &Conv3.dKernel, &Conv3.Bias, &Conv3.dBias);

			Pool2.Learning(&dConv3, &dPool2);
			ReLU2.Learning(&dPool2, &dReLU2);
			Conv2.Learning(&dReLU2, &FeatureMapL1SP, &dConv2);
			OptConv2.Updator(&Conv2.Kernel, &Conv2.dKernel, &Conv2.Bias, &Conv2.dBias);

			Pool1.Learning(&dConv2, &dPool1);
			ReLU1.Learning(&dPool1, &dReLU1);
			Conv1.Learning(&dReLU1, &TrainInput, &dConv1);
			OptConv1.Updator(&Conv1.Kernel, &Conv1.dKernel, &Conv1.Bias, &Conv1.dBias);

			Nrest = Nrest - BatchSize;
			batch++;
		}
		epoch++;
	}

	cout << "\nTraining is Done.\n";
	/*==================================================================================================================
	=                                             Test Data Evaluation			                                       =
	===================================================================================================================*/
	Matrix Test;
	Test.Mul(&TestImage, 1.0 / 255.0);


	Conv1.Inference(&Test, &FeatureMapL1);
	ReLU1.Inference(&FeatureMapL1, &FeatureMapL1ReLU);
	Pool1.Inference(&FeatureMapL1ReLU, &FeatureMapL1SP);
	Conv2.Inference(&FeatureMapL1SP, &FeatureMapL2);
	ReLU2.Inference(&FeatureMapL2, &FeatureMapL2ReLU);
	Pool2.Inference(&FeatureMapL2ReLU, &FeatureMapL2SP);
	Conv3.Inference(&FeatureMapL2SP, &FeatureMapL3);
	ReLU3.Inference(&FeatureMapL3, &FeatureMapL3ReLU);
	Flatten.Inference(&FeatureMapL3ReLU, &Flattened);
	Dense.Inference(&Flattened, &Dense1);
	Softmax.Inference(&Dense1, &Possibility);

	Matrix temp;
	float correct = 0;
	float wrong = 0;
	temp.Argmax(&Possibility, 3);

	for (int n = 0; n < TestImage.Number; n++)
	{
		if ((int)temp.Array[0][0][n][0] == TestLabel.Array[0][0][n][0])
			correct++;
		else
			wrong++;
	}

	cout << "Test Data Accuracy : " << correct / (correct + wrong) * 100 << "%\n";


	cout << "================================================================\n";
	for (int n = 4501; n < 4506; n++)
	{
		for (int r = 0; r < 28; r++)
		{
			for (int c = 0; c < 28; c++)
			{
				if (TestImage.Array[n][0][r][c] > 0) cout << "FF";
				else cout << "  ";
			}
			cout << endl;
		}
		cout << "Label : " << TestLabel.Array[0][0][n][0] << endl;
		cout << "================================================================\n";
	}

	

	Matrix TestInput, TestOutput;


	TestInput.Slice(&TestImage, 4501, 4506, 0);
	TestOutput.Slice(&TestImage, 4501, 4506, 0);
	Test.Mul(&TestInput, 1.0 / 255.0);

	Conv1.Inference(&Test, &FeatureMapL1);
	ReLU1.Inference(&FeatureMapL1, &FeatureMapL1ReLU);
	Pool1.Inference(&FeatureMapL1ReLU, &FeatureMapL1SP);
	Conv2.Inference(&FeatureMapL1SP, &FeatureMapL2);
	ReLU2.Inference(&FeatureMapL2, &FeatureMapL2ReLU);
	Pool2.Inference(&FeatureMapL2ReLU, &FeatureMapL2SP);
	Conv3.Inference(&FeatureMapL2SP, &FeatureMapL3);
	ReLU3.Inference(&FeatureMapL3, &FeatureMapL3ReLU);
	Flatten.Inference(&FeatureMapL3ReLU, &Flattened);
	Dense.Inference(&Flattened, &Dense1);
	Softmax.Inference(&Dense1, &Possibility);

	temp.Argmax(&Possibility, 3);

	temp.show_element();

	cout << "================================================================\n";

	unsigned char header[54];
	unsigned char data[28][28][3];
	ifstream input_image;
	input_image.open("9.bmp", ios::binary);
	if (!input_image.is_open()) {
		cout << "Error opening 9.bmp" << endl;
		return -1;
	}
	image Frame;
	input_image.read((char*)& Frame, sizeof(Frame));
	Matrix InputImage(1, 1, 28, 28);
	for (int r = 0; r < 28; r++)
	{
		for (int c = 0; c < 28; c++)
		{
			float x = 255.0 - (0.299 * (float)(Frame.data[27 - r][c][0]) + 0.587 * (float)(Frame.data[27 - r][c][1]) + 0.114 * (float)(Frame.data[27 - r][c][2]));
			if (x > 1) cout << "FF";
			else cout << "  ";
			InputImage.Array[0][0][r][c] = x / 255.0;
		}
		cout << endl;
	}

	Conv1.Inference(&InputImage, &FeatureMapL1);
	ReLU1.Inference(&FeatureMapL1, &FeatureMapL1ReLU);
	Pool1.Inference(&FeatureMapL1ReLU, &FeatureMapL1SP);
	Conv2.Inference(&FeatureMapL1SP, &FeatureMapL2);
	ReLU2.Inference(&FeatureMapL2, &FeatureMapL2ReLU);
	Pool2.Inference(&FeatureMapL2ReLU, &FeatureMapL2SP);
	Conv3.Inference(&FeatureMapL2SP, &FeatureMapL3);
	ReLU3.Inference(&FeatureMapL3, &FeatureMapL3ReLU);
	Flatten.Inference(&FeatureMapL3ReLU, &Flattened);
	Dense.Inference(&Flattened, &Dense1);
	Softmax.Inference(&Dense1, &Possibility);
	temp.Argmax(&Possibility, 3);
	temp.show_element();





	
	//system("pause");
	return 0;
}


/*
===================================================================
					Function Module
===================================================================
*/

void loadMNIST(Matrix* TrainImage, Matrix* TrainLabel, Matrix* TestImage, Matrix* TestLabel)
{
	fstream F_trainimage("train-images.idx3-ubyte", ios::binary | ios::in);
	for (int n = 0; n < 16; n++)
	{
		unsigned char temp = 0;
		F_trainimage.read((char*)& temp, sizeof(temp));
	}
	for (int n = 0; n < 60000; n++)
	{
		for (int r = 0; r < 28; r++)
		{
			for (int c = 0; c < 28; c++)
			{
				unsigned char temp = 0;
				F_trainimage.read((char*)& temp, sizeof(temp));
				TrainImage->Array[n][0][r][c] = (double)temp;
			}
		}
	}
	F_trainimage.close();
	//cout << "Train Image load in is done." << endl;

	fstream F_trainlabel("train-labels.idx1-ubyte", ios::binary | ios::in);
	for (int n = 0; n < 8; n++)
	{
		unsigned char temp = 0;
		F_trainlabel.read((char*)& temp, sizeof(temp));
	}
	for (int n = 0; n < 60000; n++)
	{
		unsigned char temp = 0;
		F_trainlabel.read((char*)& temp, sizeof(temp));
		TrainLabel->Array[0][0][n][0] = (double)temp;
	}
	F_trainlabel.close();
	//cout << "Train Label load in is done." << endl;

	fstream F_testimage("t10k-images.idx3-ubyte", ios::binary | ios::in);
	for (int n = 0; n < 16; n++)
	{
		unsigned char temp = 0;
		F_testimage.read((char*)& temp, sizeof(temp));
	}
	for (int n = 0; n < 10000; n++)
	{
		for (int r = 0; r < 28; r++)
		{
			for (int c = 0; c < 28; c++)
			{
				unsigned char temp = 0;
				F_testimage.read((char*)& temp, sizeof(temp));
				TestImage->Array[n][0][r][c] = (double)temp;
			}
		}
	}
	F_testimage.close();
	//cout << "Test Image load in is done." << endl;

	fstream F_testlabel("t10k-labels.idx1-ubyte", ios::binary | ios::in);
	for (int n = 0; n < 8; n++)
	{
		unsigned char temp = 0;
		F_testlabel.read((char*)& temp, sizeof(temp));
	}
	for (int n = 0; n < 10000; n++)
	{
		unsigned char temp = 0;
		F_testlabel.read((char*)& temp, sizeof(temp));
		TestLabel->Array[0][0][n][0] = (double)temp;
	}
	F_testlabel.close();
	cout << "MNIST digit data load in is done." << endl;

}

