import React from "react";
import ReactDOM from "react-dom/client";
import { PublicClientApplication } from "@azure/msal-browser";
import { MsalProvider } from "@azure/msal-react";
import App from "./App";
import { msalConfig } from "./authConfig";
import "./styles.css";

const rootElement = document.getElementById("root") as HTMLElement;
const root = ReactDOM.createRoot(rootElement);
const publicClientApplication = new PublicClientApplication(msalConfig);
void publicClientApplication.initialize();

root.render(
  <React.StrictMode>
    <MsalProvider instance={publicClientApplication}>
      <App />
    </MsalProvider>
  </React.StrictMode>,
);
