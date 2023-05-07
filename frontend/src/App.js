import { useState, useEffect, useRef } from "react";
import { BrowserRouter as Router, Route, Switch } from "react-router-dom";
import About from "./About";
import Navigation from "./Navigation";

function App() {
  const [foodInput, setFoodInput] = useState("");
  const [prediction, setPrediction] = useState("-");
  const [efScore, setEfScore] = useState("-");
  const [probability, setProbability] = useState("-");
  const debounceMs = 500;
  const debounceTimer = useRef(null);
  const [isLoading, setIsLoading] = useState(true);

  const calculateEcoScore = async (input) => {
    if (input.trim() === "") {
      setPrediction("-");
      setEfScore("-");
      setProbability("-");
      return;
    }

    const response = await fetch(`https://${process.env.REACT_APP_API_URL}/?query=${encodeURIComponent(input)}`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    });

    const { pred, ef_score, prob } = await response.json();

    setPrediction(pred);
    setEfScore(Number(ef_score).toFixed(3));
    setProbability(Number(prob).toFixed(3));
  };

  const handleFoodInputChange = (e) => {
    const input = e.target.value;
    setFoodInput(input);

    // Cancel the previous debounce if it exists
    if (debounceTimer.current !== null) {
      clearTimeout(debounceTimer.current);
    }

    // Set up a new debounce timer
    debounceTimer.current = setTimeout(() => {
      calculateEcoScore(input);
    }, debounceMs);
  };

  useEffect(() => {
    // Make a GET request to the API URL to warm up the Lambda function
    fetch(`https://${process.env.REACT_APP_API_URL}/warmup`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    })
      .then(() => setIsLoading(false));
  }, []);

  return (
    <Router>
      <>
        <header>
          <Navigation />
        </header>
        {isLoading ? (
          <div className="loading-container">
            <progress className="progress-bar" max="100"></progress>
            <p>Loading the model, please wait...</p>
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
        )}
      </>
    </Router>
  );
}

export default App;
