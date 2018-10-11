using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ML_FBLinearRegression
{
    public partial class Form1 : Form
    {
        private List<double> errors = new List<double>();

        public Form1()
        {
            InitializeComponent();
        }

        public List<double> Errors { get => errors; set => errors = value; }

        private void chart2_Click(object sender, EventArgs e)
        {

        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void Form1_Shown(object sender, EventArgs e)
        {
            for(int i =0;i<errors.Count; i++)
            {
                chart2.Series[0].Points.AddXY(i, errors[i]);
            }
        }
    }
}
