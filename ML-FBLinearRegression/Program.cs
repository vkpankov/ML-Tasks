using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;

namespace ML_FBLinearRegression
{
    class Program
    {

        static void ComputePrecMetrics(LinearRegression model, Chunk data, out double RMSE, out double R2)
        {
            double rmseSum = 0;
            double SSReg = 0, SSRes = 0;

            double targetMean = data.Y.Mean();
            for (int j = 0; j < data.X.RowCount; j++)
            {
                double target = data.Y[j];
                double predicted = model.Predict(data.X.Row(j));
                var e = Math.Pow(target - predicted, 2);
                rmseSum += e;
                SSRes += Math.Pow(target - predicted, 2);
                SSReg += Math.Pow(predicted - targetMean, 2);
            }
            R2 = SSReg / (SSReg+SSRes);
            RMSE = Math.Sqrt(rmseSum / data.X.RowCount);
        }

        static Chunk JoinChunks(List<Chunk> chunks)
        {
            Chunk resChunks = new Chunk();
            int totalRowCount = chunks.Select(m => m.X.RowCount).Sum();
            resChunks.X = Matrix<double>.Build.Dense(totalRowCount, chunks[0].X.ColumnCount);
            resChunks.Y = Vector<double>.Build.Dense(totalRowCount);
            for (int i = 0; i < chunks.Count; i++)
            {
                int curCount = chunks[i].X.RowCount;
                resChunks.X.SetSubMatrix(i * curCount, 0, chunks[i].X);
                resChunks.Y.SetSubVector(i * curCount, curCount, chunks[i].Y);
            }
            return resChunks;
        }

        static void Main(string[] args)
        {
            int featuresCount = 54;
            double learningRate = 0.00005;
            double L1Param = 0.00001;
            int itCount = 3000;
            double maxError = 0.0001;
            int chunkCount = 5;

            double[] rmseTest = new double[chunkCount];
            double[] r2Test = new double[chunkCount];
            double[] rmseTrain = new double[chunkCount];
            double[] r2Train = new double[chunkCount];
            List<double[]> chunkWeights = new List<double[]>();
            for (int i = 0; i < 54; i++)
            {
                chunkWeights.Add(new double[5]);
            }
            List<string> outputInfo = new List<string>();
            outputInfo.Add("");
            outputInfo.Add("R^2-test");
            outputInfo.Add("R^2-train");
            outputInfo.Add("RMSE-test");
            outputInfo.Add("RMSE-train");
            for (int i = 0; i < 54; i++)
                outputInfo.Add($"f{i}");


            List<Chunk> chunks = DataSetReader.LoadData("dataset.csv", featuresCount, chunkCount, true, true);
            for (int i = 0; i < chunkCount; i++)
            {
                Chunk testChunk = chunks[i];
                Chunk trainChunk = JoinChunks(chunks.Except(new Chunk[] { testChunk }).ToList());
                List<double> errors = new List<double>();
                LinearRegression model = new LinearRegression(featuresCount);
                model.Learn(trainChunk, learningRate, L1Param, maxError, itCount, ref errors);
                //Для сохранения весов в output.csv
                for (int k = 0; k < model.weights.Count; k++)
                    chunkWeights[k][i] = model.weights[k];
                ComputePrecMetrics(model, testChunk, out rmseTest[i], out r2Test[i]);
                ComputePrecMetrics(model, trainChunk, out rmseTrain[i], out r2Train[i]);
                Console.WriteLine($"T{i}: RMSETest={rmseTest[i]}, R2Test={r2Test[i]}, RMSETrain={rmseTrain[i]}, R2Train={r2Train[i]}, weights=({model.weights})");
                Form1 chartForm = new Form1();
                chartForm.Errors = errors;
                chartForm.ShowDialog();
            }

            //Запись RMSE, R2 и weights в CSV
            outputInfo[0] += ";T1;T2;T3;T4;T5;E;STD";
            for (int j = 0; j < 5; j++)
            {
                outputInfo[1] += $";{r2Test[j]}";
                outputInfo[2] += $";{r2Train[j]}";
                outputInfo[3] += $";{rmseTest[j]}";
                outputInfo[4] += $";{rmseTrain[j]}";

                for (int k = 0; k < 54; k++)
                {
                    outputInfo[5 + k] += $";{chunkWeights[k][j]}";
                }

            }
            outputInfo[1] += $";{r2Test.Mean()};{r2Test.StandardDeviation()}";
            outputInfo[2] += $";{r2Train.Mean()};{r2Train.StandardDeviation()}";
            outputInfo[3] += $";{rmseTest.Mean()};{rmseTest.StandardDeviation()}";
            outputInfo[4] += $";{rmseTrain.Mean()};{rmseTrain.StandardDeviation()}";
            for (int k = 0; k < 54; k++)
            {
                outputInfo[5 + k] += $";{chunkWeights[k].Mean()};{chunkWeights[k].StandardDeviation()}";
            }
            Console.WriteLine(outputInfo);
            File.WriteAllLines($"output.csv", outputInfo);
            //**********************

            Console.ReadKey();
        }
    }
}
