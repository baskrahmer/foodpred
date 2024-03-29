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
          "#FF8F00",
          "#FFB900",
          "#FFD54F",
          "#FFF176",
          "#ECEFF1",
          "#B2EBF2",
        ],
        borderWidth: 0,
        hoverBackgroundColor: [
          "#FFA000",
          "#FFC107",
          "#FFEB3B",
          "#FFFF8D",
          "#CFD8DC",
          "#80DEEA",
        ],
      },
    ],
  };

  const options = {
    maintainAspectRatio: false,
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
      legend: {
        position: 'right',
        labels: {
          boxWidth: 15,
          padding: 15,
          usePointStyle: true,
        },
      },
    },
  };

  return <Pie data={data} options={options} />;
};

export default EcoScorePieChart;
