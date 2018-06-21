using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace Neural_Network
{
    class NumberRecognitionDriver
    {


        
        public static void CreateAndTrain(string location)
        {
            string imageFileName = @"C:\Users\user\source\Repos\NumberRecognitionNN\Neural Network\Neural Network\train-images.idx3-ubyte";
            string labelFileName = @"C:\Users\user\source\Repos\NumberRecognitionNN\Neural Network\Neural Network\train-labels.idx1-ubyte";
            byte[] imageFile = File.ReadAllBytes(imageFileName);
            byte[] labelFile = File.ReadAllBytes(labelFileName);

            NeuralNetwork network = new NeuralNetwork(new double[784],new int[] {16,10 });

            
            int imagesIndex = 16;
            int labelIndex = 8;
            int percentDone = (int)(((double)imagesIndex / (double)imageFile.Length) * 100);

            while (imagesIndex < imageFile.Length)
            {
                double[] imageFeed = new double[784];
                for (int i = 0; i < 784; i++)
                {
                    imageFeed[i] = (imageFile[imagesIndex])/256.0;
                    imagesIndex++;
                }

                network.Feed(imageFeed);
                network.Backpropagation(NumberToDesiredOutput(labelFile[labelIndex]));
                labelIndex++;

                if(percentDone != (int)(((double)imagesIndex / (double)imageFile.Length) * 100))
                {
                    percentDone = (int)(((double)imagesIndex / (double)imageFile.Length) * 100);
                    Console.Write(percentDone + "%\n");
                }
                
            }

            network.Save(location);

            #region printregion
            /*
            int imagesIndex = 16;
            int labelIndex = 8;

            while (imagesIndex < 1000)
            {
                for (int i = 0; i < 28; i++)
                {
                    for (int j = 0; j < 28; j++)
                    {
                        if (((double)imageFile[imagesIndex]) / 256.0 > 0.5)
                        {
                            Console.Write(1 + " ");
                        }
                        else
                        {
                            Console.Write(0 + " ");
                        }
                        imagesIndex++;
                    }
                    Console.Write("\n");
                }


                Console.Write("-----" + labelFile[labelIndex] + "-----");
                labelIndex++;
                Console.Write("\n");
                Console.Write("\n");
            }
            */
            #endregion


        }

        public static void OpenAndTest(string location)
        {

            string imageFileName = @"C:\Users\user\source\Repos\NumberRecognitionNN\Neural Network\Neural Network\t10k-images.idx3-ubyte";
            string labelFileName = @"C:\Users\user\source\Repos\NumberRecognitionNN\Neural Network\Neural Network\t10k-labels.idx1-ubyte";
            byte[] imageFile = File.ReadAllBytes(imageFileName);
            byte[] labelFile = File.ReadAllBytes(labelFileName);

            NeuralNetwork network = NeuralNetwork.Load(location);
            
            int imagesIndex = 16;
            int labelIndex = 8;

            int totalImage = 0;
            int totalCorrect = 0;

            while (imagesIndex < imageFile.Length)
            {
                double[] imageFeed = new double[784];
                for (int i = 0; i < 784; i++)
                {
                    imageFeed[i] = (imageFile[imagesIndex]) / 256.0;
                    imagesIndex++;
                }
                
                //imageFeedPrint(imageFeed);
                network.Feed(imageFeed);
                //network.printNeuralNetwork();
                int answer = labelFile[labelIndex];
                int guess = getIndexOfMax(network.getOutput());
                Console.Write("answer " + answer + " guess " + guess + "\n");
                if (answer == guess)
                {
                    totalCorrect++;
                }
                totalImage++;
                labelIndex++;
            }

            Console.Write("\n"+totalCorrect+"/"+totalImage+"\n");
        }

        public static void Main()
        {
            OpenAndTest("ImageNeuralNetwork.dat");



        }

        public static double[] NumberToDesiredOutput(int desired)
        {
            double[] desiredOutput = new double[10];
            desiredOutput[desired] = 1.0;
            return desiredOutput;
        }

        public static int getIndexOfMax(double[] input)
        {
            int index = -1;
            double max = double.MinValue;
            for(int i = 0; i < input.Length; i++)
            {
                if (input[i] > max)
                {
                    index = i;
                    max = input[i];
                }
            }
            return index;
        }

        private static string arrayToString(double[] input)
        {
            string ret = "";
            for(int i = 0; i < input.Length; i++)
            {
                //ret += String.Format("{0:F2} ", input[i]);
                ret += input[i]+" ";
            }
            return ret;
        }

        private static void imageFeedPrint(double[] imageFeed)
        {
            int imageIndex = 0;
            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    if (((double)imageFeed[imageIndex]) > 0.5)
                    {
                        Console.Write(1 + " ");
                    }
                    else
                    {
                        Console.Write(0 + " ");
                    }
                    imageIndex++;
                }
                Console.Write("\n");
            }
        }
    }
}
