import React from "react";

function About() {
  return (
    <main>
      <div className="container">
        <h1>About Foodpred</h1>
        <p>
          Foodpred is a tool that allows users to estimate the ecological impact of a food product. By
          entering the name of a food product, the tool uses a language model to predict the correct category.
          This ideally helps users make fast and automated predictions about non-trivial products that are
          not present in common databases like OpenFoodFacts. For more information, visit the
          <a href="https://github.com/baskrahmer/harrygobert">source code</a>.
        </p>
        <p>
          Found a bug? Please consider reporting an issue to
          <a href="https://github.com/baskrahmer/harrygobert/issues">GitHub</a>.
        </p>
      </div>
    </main>
  );
}

export default About;
