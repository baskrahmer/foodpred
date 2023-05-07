import { useRef } from "react";

function useDebouncedCallback(callback, delay) {
  const debounceTimer = useRef(null);

  return (...args) => {
    if (debounceTimer.current !== null) {
      clearTimeout(debounceTimer.current);
    }

    debounceTimer.current = setTimeout(() => {
      callback(...args);
    }, delay);
  };
}

export default useDebouncedCallback;
