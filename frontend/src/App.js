import { useState } from "react";
import { BrowserRouter as Router, Route, Switch } from "react-router-dom";
import About from "./About";
import Navigation from "./Navigation";

function App() {
  const [foodInput, setFoodInput] = useState("");
  const [prediction, setPrediction] = useState("-");
  const [efScore, setEfScore] = useState("-");
  const [probability, setProbability] = useState("-");

  const calculateEcoScore = async () => {
    if (foodInput.trim() === "") return;

    const response = await fetch(`https://${process.env.REACT_APP_API_URL}/?query=${encodeURIComponent(foodInput)}`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    });

    const { pred, ef_score, prob } = await response.json();

    setPrediction(pred);
    setEfScore(ef_score);
    setProbability(prob);
  };

  const handleFoodInputChange = (e) => {
    setFoodInput(e.target.value);
    calculateEcoScore();
  };

  return (
    <Router>
      <>
        <header>
          <Navigation />
        </header>
        <Switch>
          <Route path="/" exact>
            <main>
              <div className="container">
                <h1>EcoScore Calculator</h1>
                <h1>{process.env.REACT_APP_API_URL}</h1>
                <div className="input-container">
                  <input
                    type="text"
                    id="foodInput"
                    placeholder="Enter food product"
                    value={foodInput}
                    onChange={handleFoodInputChange}
                  />
                  <p className="result" id="prediction">
                    Prediction: <span id="predictionValue">{prediction}</span>
                  </p>
                  <p className="result" id="efScore">
                    EF Score: <span id="efScoreValue">{efScore}</span>
                  </p>
                  <p className="result" id="probability">
                    Probability: <span id="probabilityValue">{probability}</span>
                  </p>
                </div>
              </div>
            </main>
          </Route>
          <Route path="/about">
            <About />
          </Route>
        </Switch>
      </>
    </Router>
  );
}

export default App;
