import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import VideoAnalyzer from './components/VideoAnalyzer';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={
          <Layout>
            <VideoAnalyzer />
          </Layout>
        } />
      </Routes>
    </Router>
  );
}

export default App;