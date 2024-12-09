document.getElementById("projection-form").addEventListener("submit", async function(event) {
    event.preventDefault();

    const playerName = document.getElementById("player_name").value;

    if (!playerName) {
        alert("Please enter a player name.");
        return;
    }

    try {
        const response = await fetch("/player_projection", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ player_name: playerName })
        });

        const data = await response.json();
        const resultsDiv = document.getElementById("results");
        const messageDiv = document.getElementById("message");
        const resultGraph = document.getElementById("result_graph");
        const mseGraph = document.getElementById("mse_graph");
        const maeGraph = document.getElementById("mae_graph");

        if (response.ok) {
            // Display the message from the server
            messageDiv.innerText = data.message;

            // Update MSE graph
            if (data.mse_graph_url) {
                mseGraph.src = `${data.mse_graph_url}?t=${new Date().getTime()}`;
                mseGraph.style.display = "block";
            }

            // Update MAE graph
            if (data.mae_graph_url) {
                maeGraph.src = `${data.mae_graph_url}?t=${new Date().getTime()}`;
                maeGraph.style.display = "block";
            }
        } else {
            messageDiv.innerText = data.error || "An error occurred.";
            resultGraph.style.display = "none";
            mseGraph.style.display = "none";
            maeGraph.style.display = "none";
        }

        resultsDiv.style.display = "block";
    } catch (error) {
        console.error("Error:", error);
        alert("Failed to communicate with the server.");
    }
});
