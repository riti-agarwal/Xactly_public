import React, { createContext, useState, useContext } from 'react';

// Create a context for sidebar visibility
const SidebarContext = createContext();

// Custom hook to use the sidebar context
export const useSidebar = () => useContext(SidebarContext);

// Provider component
export const SidebarProvider = ({ children }) => {
  const [isSidebarVisible, setIsSidebarVisible] = useState(false);

  const toggleSidebar = () => {
    setIsSidebarVisible(prev => !prev);
  };

  return (
    <SidebarContext.Provider value={{ isSidebarVisible, toggleSidebar }}>
      {children}
    </SidebarContext.Provider>
  );
};
