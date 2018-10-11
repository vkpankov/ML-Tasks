using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML_FBLinearRegression
{
    static class DataSetReader
    {
        private static Matrix<double> Standardize(Matrix<double> x)
        {
            Matrix<double> normalizedMatrix = Matrix<double>.Build.Dense(x.RowCount, x.ColumnCount);
            for (int i = 0; i < x.ColumnCount; i++)
            {
                var mean = x.Column(i).Mean();
                var stddev = x.Column(i).StandardDeviation();
                if (stddev == 0)
                    continue;
                var normalizedColumn = x.Column(i).Subtract(mean) * 1.0 / stddev;
                normalizedMatrix.SetColumn(i, normalizedColumn);
            }
            return normalizedMatrix;

        }
        private static void Shuffle(ref Matrix<double> data)
        {
            Random rng = new Random();
            int n = data.RowCount;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                var value = data.Row(k);
                data.SetRow(k, data.Row(n));
                data.SetRow(n, value);
            }
        }
        private static List<Chunk> GetChunks(Matrix<double> data, int chunkCount, bool standardize)
        {
            Matrix<double> x = data.SubMatrix(0, data.RowCount, 0, data.ColumnCount - 1);
            Vector<double> y = data.SubMatrix(0, data.RowCount, data.ColumnCount - 1, 1).Column(0);
            List<Chunk> chunks = new List<Chunk>();
            int chunkSize = x.RowCount / chunkCount;

            if (standardize)
                x = Standardize(x);

            for (int i = 0; i < chunkCount; i++)
            {
                var curX = x.SubMatrix(i * chunkSize, chunkSize, 0, x.ColumnCount);
                var curY = y.SubVector(i * chunkSize, chunkSize);
                chunks.Add(new Chunk { X = curX, Y = curY });
            }
            return chunks;
        }

        public static List<Chunk> LoadData(string fileName, int featureCount, int chunkCount, bool standardize = true, bool shuffle = true)
        {
            List<string> inputData = File.ReadAllLines(fileName).ToList();
            Matrix<double> data = Matrix<double>.Build.Dense(inputData.Count, featureCount + 1);
            for (int i = 0; i < inputData.Count; i++)
            {

                var curRow = inputData[i].Trim('"').Split(';','\t').Select(rowStr => Double.Parse(rowStr.Replace(".",","))).ToArray();
                for (int j = 0; j < curRow.Length; j++)
                {
                    data[i, j] = curRow[j];
                }
            }
            if (shuffle)
            {
                Shuffle(ref data);
            }
            return GetChunks(data, chunkCount, standardize);
        }
    }
}
