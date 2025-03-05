document.addEventListener("DOMContentLoaded", function () {
    const searchInput = document.getElementById("searchInput");
    const searchBtn = document.getElementById("searchBtn");
    const suggestionsBox = document.getElementById("suggestionsBox");
    const recentSearchesList = document.getElementById("recentSearchesList");

    // Load recent searches from localStorage
    function loadRecentSearches() {
        let searches = JSON.parse(localStorage.getItem("recentSearches")) || [];
        recentSearchesList.innerHTML = searches.map(search => `
            <div class="suggestion-item" onclick="selectSearch('${search}')">
                <i class="fas fa-history"></i> ${search}
            </div>
        `).join("");
        suggestionsBox.style.display = searches.length ? "block" : "none";
    }

    // Save search to localStorage
    function saveSearch(query) {
        let searches = JSON.parse(localStorage.getItem("recentSearches")) || [];
        if (!searches.includes(query)) {
            searches.unshift(query); // Add new search at the top
            if (searches.length > 5) searches.pop(); // Keep only 5 recent searches
            localStorage.setItem("recentSearches", JSON.stringify(searches));
        }
        loadRecentSearches();
    }

    function performSearch() {
        const keyword = searchInput.value.trim();
        if (keyword !== "") {
            saveSearch(keyword);
            window.location.href = `/results?keyword=${encodeURIComponent(keyword)}`;
        }
    }

    // Event Listeners for Search
    searchBtn.addEventListener("click", performSearch);
    searchInput.addEventListener("keypress", function (event) {
        if (event.key === "Enter") {
            performSearch();
        }
    });

        // Ensure sentimentCounts and wordFrequencies are defined
        if (typeof sentimentCounts !== "undefined" && typeof wordFrequencies !== "undefined") {
            // Ensure sentimentChart element exists before creating the chart
            var chartElement = document.getElementById("sentimentChart");
            if (chartElement) {
                var ctx = chartElement.getContext("2d");
                var sentimentChart = new Chart(ctx, {
                    type: "pie",
                    data: {
                        labels: ["Neutral", "Negative", "Positive", "Mixed"],
                        datasets: [{
                            data: [
                                sentimentCounts[0] || 0,  // Neutral
                                sentimentCounts[1] || 0,  // Negative
                                sentimentCounts[2] || 0,  // Positive
                                sentimentCounts[3] || 0   // Mixed
                            ],
                            backgroundColor: [
                                "#007bff",  // Neutral (Blue)
                                "#ff6384",  // Negative (Red)
                                "#4bc0c0",  // Positive (Green)
                                "#6c757d"   // Mixed (Gray)
                            ],
                            hoverBackgroundColor: [
                                "#66b0ff",  // Lighter blue
                                "#ff8099",  // Lighter red
                                "#74d4d4",  // Lighter green
                                "#95a5a6"   // Lighter gray
                            ]
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: "top",
                                labels: {
                                    font: {
                                        size: 14
                                    },
                                    color: "#333"
                                }
                            }
                        }
                    }
                });
            }
        }


        
    // Select a recent search
    window.selectSearch = function (query) {
        searchInput.value = query;
    };

    // Load recent searches on page load
    loadRecentSearches();


    
});
