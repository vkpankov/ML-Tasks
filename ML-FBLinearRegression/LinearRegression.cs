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

        public void Learn(Chunk data, double learningRate, double L1RegParam, double maxError, int maxIterationCount, ref List<double> errors, Chunk testData)
        {
            var m = data.X.RowCount;
            var featuresCount = weights.Count;
            int it = 3;
            double error = 1;
            do
            {
                double rmse = 1;
                Vector<double> wGrad = Vector<double>.Build.Dense(weights.Count);
                double bGrad = 0;
                var e = data.Y - this.Predict(data.X);
                for (int i = 0; i < data.X.ColumnCount; i++)
                    wGrad[i] += e * data.X.Column(i);
                bGrad = e.Sum() * -2.0 / m;
                wGrad *= -2.0 / m;

                weights -= wGrad * learningRate;
                for(int k=0;k<weights.Count; k++)
                    weights[k] -= Math.Min(weights[k], (Math.Sign(weights[k]) * L1RegParam));

                b -= bGrad * learningRate;
                rmse = Math.Sqrt(e.PointwisePower(2).Sum() / data.Y.Count);
                error = rmse;
                errors.Add(rmse);
               
      
                    it++;
            } while (it < maxIterationCount && error > maxError);
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
