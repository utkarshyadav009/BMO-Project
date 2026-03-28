// Hijack getGamepads to unconditionally return an empty array
try {
    Object.defineProperty(navigator, 'getGamepads', {
        value: function() { return []; },
        configurable: true,
        writable: true
    });
} catch(e) {
    // Fallback if the browser locks the object
    navigator.getGamepads = function() { return []; };
}

// Catch the old webkit prefix just in case Emscripten looks for it
try {
    Object.defineProperty(navigator, 'webkitGetGamepads', {
        value: function() { return []; },
        configurable: true,
        writable: true
    });
} catch(e) {}
