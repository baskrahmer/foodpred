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
          not present in common databases like OpenFoodFacts. For more information and technical details, visit the <a href="https://github.com/baskrahmer/harrygobert">source code</a>.
          Found a bug? Please consider reporting an issue to <a href="https://github.com/baskrahmer/harrygobert/issues">GitHub</a>.
        </p>
        <br/>
        <h1>Limitations</h1>
        <p>
          Foodpred is trained on all OpenFoodFacts products for which a CIQUAL classification is known. This
          data is skewed towards the French language and contains products names which may taint the classification.
          A good example of this is fruit names: <a href="https://world.openfoodfacts.org/cgi/search.pl?search_terms=Mango&search_simple=1&action=process">there are many Mango-flavored products in the database</a>,
          which results in a query of just "Mango" to be skewed towards processed foods containing this product.
          For real-world applications, these biases need to be carefully considered and a custom model should be created.
        </p>
      </div>
    </main>
  );
}

export default About;
