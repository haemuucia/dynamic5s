import { initializeApp } from "https://www.gstatic.com/firebasejs/11.8.1/firebase-app.js";
import { getAnalytics } from "https://www.gstatic.com/firebasejs/11.8.1/firebase-analytics.js";
  // TODO: Add SDKs for Firebase products that you want to use
  // https://firebase.google.com/docs/web/setup#available-libraries

  // Your web app's Firebase configuration
  // For Firebase JS SDK v7.20.0 and later, measurementId is optional
  const firebaseConfig = {
    apiKey: "AIzaSyCvaTceOrpkFsc9yfm8SesDP61AGLFQ2dk",
    authDomain: "slab-7e423.firebaseapp.com",
    projectId: "slab-7e423",
    storageBucket: "slab-7e423.firebasestorage.app",
    messagingSenderId: "236607768404",
    appId: "1:236607768404:web:d50372aefaaa4ce0ffeac1",
    measurementId: "G-J6RFN6BFL9"
  };

  // Initialize Firebase
  const app = initializeApp(firebaseConfig);
  const analytics = getAnalytics(app);
