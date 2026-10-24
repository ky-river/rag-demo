window.RAG_CONFIG = window.RAG_CONFIG || {
  apiBaseUrl: `${window.location.protocol}//${window.location.hostname}:8000`,
};

window.getRagApiBaseUrl = function getRagApiBaseUrl() {
  const fallback = `${window.location.protocol}//${window.location.hostname}:8000`;
  return (window.RAG_CONFIG.apiBaseUrl || fallback).replace(/\/+$/, "");
};
