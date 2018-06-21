using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.Serialization;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace Neural_Network
{
    [Serializable]
    class Block:ISerializable
    {
        double[] inputMatrix;
        double[] rawOutputMatrix;
        double[] outputMatrix;
        double[,] connectionMatrix;
        double[] biasMatrix;

        public Block(double[] inputMatrix,int outputSize)
        {
            this.inputMatrix = inputMatrix;
            connectionMatrix = GenerateRandomConnection(inputMatrix.GetLength(0),outputSize);
            biasMatrix = GenerateRandomBias(outputSize);
            outputMatrix = new double[outputSize];
            rawOutputMatrix = new double[outputSize];
            GenerateOutput();

        }

        public Block generateNextBlock(int outputSize)
        {
            Block nextBlock = new Block(outputMatrix,outputSize);
            return nextBlock;
        }

        public void GenerateOutput()
        {

            for(int i = 0; i < connectionMatrix.GetLength(0); i++)
            {
                double outElement = 0;
                for(int j = 0; j < connectionMatrix.GetLength(1); j++)
                {
                    outElement += (connectionMatrix[i, j] * inputMatrix[j]);
                }
                rawOutputMatrix[i] = outElement + biasMatrix[i];
                outputMatrix[i] = squishificationFuction(rawOutputMatrix[i]);
            }
            
        }

        public void Input(double[] input)
        {
            inputMatrix = input;
        }

        public void printBlock()
        {
            for(int i = 0; i < inputMatrix.Length; i++)
            {
                Console.Write(String.Format("{0:F2} ",inputMatrix[i]));
            }
            Console.Write("\n\n");

            /*
            for(int i = 0; i < connectionMatrix.GetLength(0); i++)
            {
                for(int j = 0; j < connectionMatrix.GetLength(1); j++)
                {
                    Console.Write(String.Format("{0:F2} ", connectionMatrix[i, j]));
                }
                Console.Write("\n");
            }
            Console.Write("\n\n");
            */
            for (int i = 0; i < biasMatrix.Length; i++)
            {
                Console.Write(String.Format("{0:F2} ", biasMatrix[i]));
            }
            Console.Write("\n\n");
            
            
            for (int i = 0; i < rawOutputMatrix.Length; i++)
            {
                Console.Write(String.Format("{0:F2} ", rawOutputMatrix[i]));
            }
            Console.Write("\n\n");
            

            for (int i = 0; i < outputMatrix.Length; i++)
            {
                Console.Write(String.Format("{0:F2} ", outputMatrix[i]));
            }
            Console.Write("\n\n");
        }


        public double[] Backpropagation(double[] desiredOutput)
        {
            double[] desiredInput = new double[inputMatrix.Length];
            for(int i = 0; i < outputMatrix.Length; i++)
            {
                //delta C / delta B
                double dcdb = squishificationFuctionDerivative(rawOutputMatrix[i]) * 2 * (outputMatrix[i] - desiredOutput[i]);

                for (int j = 0; j < inputMatrix.Length; j++)
                {
                    //delta C / delta W
                    double dcdw = inputMatrix[j] * squishificationFuctionDerivative(rawOutputMatrix[i]) * 2 * (outputMatrix[i] - desiredOutput[i]);


                    //delta C / delta A(-1)
                    double dcda = connectionMatrix[i, j] * squishificationFuctionDerivative(rawOutputMatrix[i]) * 2 * (outputMatrix[i] - desiredOutput[i]);


                    //Apply deltas
                    connectionMatrix[i, j] -= dcdw;

                    desiredInput[j] -= dcda;

                }
                biasMatrix[i] -= dcdb;
            }
            for (int i = 0; i < inputMatrix.Length; i++)
            {
                desiredInput[i] = desiredInput[i] + inputMatrix[i];
            }


            return desiredInput;
        }

        public double[] getOutput()
        {
            return outputMatrix;
        }

        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            info.AddValue("inputMatrix", inputMatrix);
            info.AddValue("rawOutputMatrix", rawOutputMatrix);
            info.AddValue("outputMatrix", outputMatrix);
            info.AddValue("connectionMatrix", connectionMatrix);
            info.AddValue("biasMatrix", biasMatrix);
        }

        public Block(SerializationInfo info, StreamingContext context)
        {
            inputMatrix = (double[])info.GetValue("inputMatrix", typeof(double[]));
            rawOutputMatrix = (double[])info.GetValue("rawOutputMatrix", typeof(double[]));
            outputMatrix = (double[])info.GetValue("outputMatrix", typeof(double[]));
            connectionMatrix = (double[,])info.GetValue("connectionMatrix", typeof(double[,]));
            biasMatrix = (double[])info.GetValue("biasMatrix", typeof(double[]));
        }



        //private methods


        public static double squishificationFuction(double input)
        {
            return 1.0/(1.0+(Math.Exp(-input)));
            //return Math.Log(1 + Math.Exp(input));
        }
        
        
        public static double squishificationFuctionDerivative(double input)
        {
            return squishificationFuction(input)*(1 - squishificationFuction(input));
            //return 1.0 / (1.0 + Math.Exp(-input));
        }

        private static double[,] GenerateRandomConnection(int inputSize, int outputSize)
        {
            double[,] connection = new double[outputSize,inputSize];
            for(int i = 0;i< outputSize;i++)
            {
                for(int j = 0; j < inputSize; j++)
                {
                    connection[i,j] = NeuralNetwork.RandomWeight();
                }
            }
            return connection;
        }

        private static double[] GenerateRandomBias(int outputSize)
        {
            double[] bias = new double[outputSize];
            for(int i = 0; i < outputSize; i++)
            {
                bias[i] = NeuralNetwork.RandomWeight();
            }
            return bias;
        }

    }

}
 