using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML_FBLinearRegression
{
    class LinearRegression
    {
        public Vector<double> weights;
        public double b;

        public LinearRegression(int featuresCount)
        {
            weights = Vector<double>.Build.Random(featuresCount);
            b = new Random().NextDouble();
        }

        public void Learn(Chunk data, double learningRate, double L1RegParam, double maxError, int maxIterationCount, ref List<double> errors)
        {
            var m = data.X.RowCount;
           
            var featuresCount = weights.Count;

            int i = 1;
            double error = 1;
            Vector<double> curWeights = Vector<double>.Build.Dense(weights.Count);
            do
            {
                double rmse = 100;
                for (int j = 0; j < featuresCount; j++)
                {
                    Vector<double> featureX = data.X.Column(j);
                    var e = data.Y - ((weights[j] * featureX).Add(b));
                    curWeights[j] = weights[j] - learningRate * (-2.0 / m) * e * featureX + L1RegParam * Math.Sign(weights[j]);
                    b = b - learningRate * (-2.0 / m) * e.Sum();
                }
                error = (curWeights - weights).L2Norm();
                curWeights.CopyTo(weights);

                var yDiff = this.Predict(data.X) - data.Y;
                rmse = Math.Sqrt(yDiff.PointwisePower(2).Sum() / data.Y.Count);
                errors.Add(rmse);
                i++;
            } while (i < maxIterationCount && error > maxError);
        }

        public double Predict(Vector<double> x)
        {
            return weights * x + b;
        }
        public Vector<double> Predict(Matrix<double> x)
        {
            var y = Vector<double>.Build.Dense(x.RowCount);
           
            for(int i =0; i<x.RowCount; i++)
            {
                y[i] = Predict(x.Row(i));
            }
            return y;
        }
    }
}
