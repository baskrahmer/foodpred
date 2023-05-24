import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Route, Switch } from "react-router-dom";
import About from "./About";
import Navigation from "./Navigation";
import EcoScorePieChart from "./EcoScorePieChart";
import useDebouncedCallback from "./DebouncedCallback";

function App() {
  const [foodInput, setFoodInput] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [efScore, setEfScore] = useState(null);
  const [co2, setCo2] = useState(null);
  const [probability, setProbability] = useState(null);
  const debounceMs = 500;
  const [isLoading, setIsLoading] = useState(true);
  const [efPhases, setEfPhases] = useState({});

  const calculateEcoScore = async (input) => {
    if (input.trim() === "") {
      setPrediction(null);
      setEfScore(null);
      setCo2(null);
      setProbability(null);
      return;
    }

    const response = await fetch(
      `https://${process.env.REACT_APP_API_URL}/?query=${encodeURIComponent(input)}`,
      {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      }
    );

    const { pred, ef_score, prob, ef_phases, co2 } = await response.json();

    setPrediction(pred);
    setEfScore(Number(ef_score).toFixed(3));
    setCo2(Number(co2).toFixed(3));
    setProbability(Number(prob).toFixed(3));
    setEfPhases(ef_phases);
  };

  const debouncedCalculateEcoScore = useDebouncedCallback(calculateEcoScore, debounceMs);

  const handleFoodInputChange = (e) => {
    const input = e.target.value;
    setFoodInput(input);
    debouncedCalculateEcoScore(input);
  };

  useEffect(() => {
    fetch(`https://${process.env.REACT_APP_API_URL}/warmup`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    })
      .then(() => setIsLoading(false))
      .catch((error) => {
        console.error("Error warming up the Lambda function:", error);
        setIsLoading(false);
      });
  }, []);

  return (
    <Router>
      <React.Fragment>
        <header>
          <Navigation />
        </header>
        {isLoading ? (
          <div className="loading-container">
            <progress className="progress-bar" max="100"></progress>
            <p>Loading the model, this can take up to a minute.</p>
          </div>
        ) : (
          <Switch>
            <Route path="/" exact>
              <main>
                <div className="container">
                  <h1>Food Predictor</h1>
                  <div className="input-container">
                    <input
                      type="text"
                      id="foodInput"
                      placeholder="Enter food product"
                      value={foodInput}
                      onChange={handleFoodInputChange}
                    />
                  </div>
                  {foodInput.trim() !== "" && (
                    <div className="output-container">
                      <div className="result-container">
                        <table>
                          <tbody>
                            <tr>
                              <td>Prediction</td>
                              <td><span id="predictionValue">{prediction}</span></td>
                            </tr>
                            <tr>
                              <td>kg Co2</td>
                              <td><span id="co2Value">{co2}</span></td>
                            </tr>
                            <tr>
                              <td>EF Score</td>
                              <td><span id="efScoreValue">{efScore}</span></td>
                            </tr>
                            <tr>
                              <td>Probability</td>
                              <td><span id="probabilityValue">{probability}</span></td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                      <div className="chart-container">
                        <EcoScorePieChart efPhases={efPhases} />
                      </div>
                    </div>
                  )}
                </div>
              </main>
            </Route>
            <Route path="/about">
              <About />
            </Route>
          </Switch>
        )}
      </React.Fragment>
    </Router>
  );
}

export default App;
