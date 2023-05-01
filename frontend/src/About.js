import React from "react";

function About() {
  return (
    <main>
      <div className="container">
        <h1>About Foodpred</h1>
        <p>
          Foodpred is a tool that allows users to estimate the ecological footprint of a food product. By
          entering the name of a food product, the tool uses a language model to predict the correct category.
          This ideally helps users make fast and automated predictions about non-trivial products that are
          not present in common databases like OpenFoodFacts.
        </p>
      </div>
    </main>
  );
}

export default About;
