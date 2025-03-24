document.addEventListener("DOMContentLoaded", function () {
    const searchInput = document.getElementById("searchInput");
    const searchBtn = document.getElementById("searchBtn");
    const suggestionsBox = document.getElementById("suggestionsBox");
    const recentSearchesList = document.getElementById("recentSearchesList");

    // Create an error message container
    const errorMessage = document.createElement("div");
    errorMessage.id = "errorMessage";
    errorMessage.style.color = "red";
    errorMessage.style.fontSize = "14px";
    errorMessage.style.marginTop = "5px";
    searchInput.parentElement.appendChild(errorMessage); // Add it below the search bar

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

    // Select a recent search and trigger search
    window.selectSearch = function (query) {
        searchInput.value = query;
        performSearch(query); // Automatically trigger search
    };

    // Function to validate input
    function isValidSearch(query) {
        const regex = /^[A-Za-z\s]+$/; // Allows only letters and spaces
        return regex.test(query) && query.length > 1; // At least 2 characters
    }

    // Function to perform search with validation
    function performSearch(query) {
        if (!query) return;

        if (!isValidSearch(query)) {
            searchInput.classList.add("is-invalid"); // Add Bootstrap's invalid class
            errorMessage.innerText = "That doesn't seem like a valid name. Avoid numbers and symbols.";
            return;
        }

        // If valid, remove error styles
        searchInput.classList.remove("is-invalid");
        errorMessage.innerText = "";

        saveSearch(query);
        window.location.href = `/results?keyword=${encodeURIComponent(query)}`;
    }

    // Save search to localStorage
    function saveSearch(query) {
        let searches = JSON.parse(localStorage.getItem("recentSearches")) || [];
        if (!searches.includes(query)) {
            searches.unshift(query); // Add new search at the top
            if (searches.length > 3) searches.pop(); 
            localStorage.setItem("recentSearches", JSON.stringify(searches));
        }
        loadRecentSearches();
    }

    // Event Listeners for Search
    searchBtn.addEventListener("click", function () {
        performSearch(searchInput.value.trim());
    });

    searchInput.addEventListener("keypress", function (event) {
        if (event.key === "Enter") {
            performSearch(searchInput.value.trim());
        }
    });

    // Remove error message on input change
    searchInput.addEventListener("input", function () {
        searchInput.classList.remove("is-invalid");
        errorMessage.innerText = "";
    });

    // Load recent searches on page load
    loadRecentSearches();
});
