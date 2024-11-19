/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'space-dark': '#0a192f',
        'space-light': '#112240',
        'space-cyan': '#64ffda',
        'space-text': '#ccd6f6',
        'space-text-secondary': '#8892b0',
      },
    },
  },
  plugins: [],
}

