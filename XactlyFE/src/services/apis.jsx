/**
 * API service for handling backend requests
 */

/**
 * Fetches random product images with quality metrics from the backend
 * @param {number} count - Number of images to fetch (default: 20)
 * @returns {Promise} - Promise that resolves to an array of product objects
 */
export const fetchRandomImages = async () => {
  const response = await fetch(`http://127.0.0.1:5000/random_images`);

  if (!response.ok) {
    throw new Error(`HTTP error! Status: ${response.status}`);
}

  const data = await response.json();
  return data;
};
