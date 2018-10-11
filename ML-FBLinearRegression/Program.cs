using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math.Optimization;
using Accord.Math.Optimization.Losses;
using Accord.Statistics.Kernels;
using Accord.Statistics.Models.Regression.Fitting;
using Accord.Statistics.Models.Regression.Linear;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;

namespace ML_FBLinearRegression
{
    class Program
    {
        static Vector<double> Normalize(Vector<double> x)
        {
            return x.Subtract(x.Mean()) * 1.0 / x.StandardDeviation();
        }


        static double EmbTest(Matrix<double> features, int featuresCount, out double[] weights, out double b)
        {
            MultivariateLinearRegression mvlr = new MultivariateLinearRegression();
            
            var ols = new OrdinaryLeastSquares()
            {
                UseIntercept = true,
                IsRobust = true
            };
            var x = features.SubMatrix(0, features.RowCount, 0, featuresCount).ToRowArrays();
            var y = features.SubMatrix(0, features.RowCount, featuresCount, 1).ToColumnArrays()[0];
            MultipleLinearRegression mlr = ols.Learn(x, y);

            double RMSE = 0;
            for (int j = 0; j < features.RowCount; j++)
            {
                var curRow = features.Row(j);
                double realY = curRow[featuresCount];
                var featureCol = curRow.SubVector(0, featuresCount);
                double yp = mlr.Transform(featureCol.ToArray());
                var e = Math.Abs(realY - yp);
                if (e > 100)
                {

                }
                RMSE += e;
            }
            weights = mlr.Weights;
            b = mlr.Transform(new double[54]);
            return RMSE / features.RowCount;
        }



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
                rmseSum += Math.Sqrt(e);
                SSRes += Math.Pow(target - predicted, 2);
                SSReg += Math.Pow(predicted - targetMean, 2);
            }
            R2 = SSReg / (SSReg+SSRes);
            RMSE = rmseSum / data.X.RowCount;
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
            double learningRate = 0.0001;
            int itCount = 5000;
            int chunkCount = 1;

            List<Chunk> chunks = DataSetReader.LoadData("dataset.csv", featuresCount, chunkCount, true, true);

            List<double> errors = new List<double>();
            LinearRegression model = new LinearRegression(featuresCount);

            Chunk toTrain = JoinChunks(chunks.ToList());

            model.Learn(toTrain, learningRate, 0, 0.0001, itCount, ref errors);

            Form1 chartForm = new Form1();
            chartForm.Errors = errors;
            chartForm.ShowDialog();

            double rmse, r2;
            ComputePrecMetrics(model, chunks[0], out rmse, out r2);



        }
    }
}
