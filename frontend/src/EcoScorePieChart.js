import React from "react";
import { Pie } from "react-chartjs-2";
import "./EcoScorePieChart.css";
import { Chart, ArcElement, Tooltip, Legend } from 'chart.js';
Chart.register(ArcElement, Tooltip, Legend);

const EcoScorePieChart = ({ efPhases }) => {
  const labels = Object.keys(efPhases).map(label => {
    return label.charAt(0).toUpperCase() + label.slice(1);
  });
  const values = Object.values(efPhases);
  const total = values.reduce((acc, val) => acc + val, 0);

  const data = {
    labels,
    datasets: [
      {
        data: values,
        backgroundColor: [
          "#FF6384",
          "#36A2EB",
          "#FFCE56",
          "#4BC0C0",
          "#E7E9ED",
          "#9966FF",
        ],
      },
    ],
  };

  const options = {
    plugins: {
      tooltip: {
        callbacks: {
          title: (context) => {
            return context[0].label;
          },
          label: (context) => {
            const value = context.parsed;
            const percentage = ((value / total) * 100).toFixed(2);
            return `${percentage}%`;
          },
        },
      },
    },
  };

  return <Pie data={data} options={options} />;
};

export default EcoScorePieChart;
