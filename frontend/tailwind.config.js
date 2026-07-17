/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],

  darkMode: "class",

  theme: {
    extend: {

      fontFamily: {
        sans: ["Inter", "Segoe UI", "sans-serif"],
      },

      colors: {

        primary: {
          50: "#eef7ff",
          100: "#d8ebff",
          200: "#b6dbff",
          300: "#84c3ff",
          400: "#4da4ff",
          500: "#1677ff",
          600: "#005ee6",
          700: "#004bb5",
          800: "#003a8c",
          900: "#00275e",
        },

        success: "#10b981",
        warning: "#f59e0b",
        danger: "#ef4444",
        info: "#3b82f6",

      },

      borderRadius: {
        xl: "16px",
        "2xl": "22px",
        "3xl": "30px",
      },

      boxShadow: {
        soft: "0 8px 30px rgba(0,0,0,0.08)",
        card: "0 15px 35px rgba(0,0,0,0.08)",
        glow: "0 0 30px rgba(22,119,255,.25)",
      },

      animation: {
        fade: "fade .5s ease",
        float: "float 5s ease-in-out infinite",
        pulseSlow: "pulse 3s infinite",
      },

      keyframes: {
        fade: {
          "0%": {
            opacity: 0,
            transform: "translateY(20px)"
          },
          "100%": {
            opacity: 1,
            transform: "translateY(0)"
          }
        },

        float: {
          "0%,100%": {
            transform: "translateY(0px)"
          },
          "50%": {
            transform: "translateY(-10px)"
          }
        }
      }
    },
  },

  plugins: [],
}