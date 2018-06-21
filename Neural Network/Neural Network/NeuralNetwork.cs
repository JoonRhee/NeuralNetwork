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
    class NeuralNetwork : ISerializable
    {
        List<Block> blocks = new List<Block>();
        public NeuralNetwork(double[] input,int[] layers)
        {
            blocks.Add(new Block(input,layers[0]));
            for(int i = 1; i < layers.Length; i++)
            {
                blocks.Add(blocks[i - 1].generateNextBlock(layers[i]));
            }
        }

        public void Feed(double[] input)
        {
            blocks[0].Input(input);
            foreach(Block block in blocks)
            {
                block.GenerateOutput();
            }
        }



        public void printNeuralNetwork()
        {
            int count = 1;
            foreach (Block block in blocks)
            {
                Console.Write("Block#" + count +"\n");
                block.printBlock();

                Console.Write("\n\n\n\n\n");
                count++;
            }
        }

        public double[] Backpropagation(double[] desiredOutput)
        {
            
            for(int i = blocks.Count - 1; i >= 0; i--)
            {
                desiredOutput = blocks[i].Backpropagation(desiredOutput);
                NeuralNetwork.PrintArray(desiredOutput);
            }
            return desiredOutput;
        }

        public string Save(string location)
        {
            int counter = 0;
            string countLoc = location;
            while (File.Exists(countLoc))
            {
                countLoc = location.Insert(location.LastIndexOf("."), ("" +counter));
                counter++;
            }
            location = countLoc;


            Stream stream = File.Open(location, FileMode.CreateNew);

            BinaryFormatter bf = new BinaryFormatter();

            bf.Serialize(stream, this);
            stream.Close();

            return location;
        }

        public static NeuralNetwork Load(string location)
        {
            Stream stream = File.Open(location, FileMode.Open);

            BinaryFormatter bf = new BinaryFormatter();

            NeuralNetwork saved = (NeuralNetwork)bf.Deserialize(stream);
            stream.Close();

            return saved;

        }

        public double[] getOutput()
        {
            return blocks[blocks.Count - 1].getOutput();
        }
















        
        private static Random rand = new Random();

        public static void PrintArray(double[] array)
        {
            for(int i = 0; i < array.Length; i++)
            {
                Console.Write(String.Format("{0:F2} ",array[i]));
            }
            Console.Write("\n");
        }

        public static void PrintArray(double[,] array)
        {
            for (int i = 0; i < array.GetLength(0); i++)
            {
                for(int j =0;j< array.GetLength(1); j++)
                {
                    Console.Write(array[i,j] + "\t");
                }
                Console.Write("\n");
            }
        }

        public static double RandomWeight()
        {
            return (double)rand.NextDouble()*2 -1;
        }


        /*
        public static void Main()
        {
            NeuralNetwork nn = new NeuralNetwork(new double[]{1f,2f,3f,4f,5f,6f },new int[]{4,16,16,16,16,10 });
            nn.printNeuralNetwork();

            string newloc = nn.Save("NeuralNetwork.dat");
            NeuralNetwork saved = NeuralNetwork.Load(newloc);
            saved.printNeuralNetwork();
        }
        */





        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            info.AddValue("Blocks",blocks);
        }

        public NeuralNetwork(SerializationInfo info, StreamingContext context)
        {
            blocks = (List<Block>)info.GetValue("Blocks",typeof(List<Block>));
        }
    }
}
