// API Client for Headspace - Direct to Supabase

import {
  createClient,
} from "https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm";

// --- Config ---
const SUPABASE_URL = "https://pwxleudvclhcbksvjcho.supabase.co";
const SUPABASE_ANON_KEY =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB3eGxldWR2Y2xoY2Jrc3ZqY2hvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDAyNTI4ODksImV4cCI6MjAxNTgyODg4OX0.pLp6NB23D1d_v4i21xI433Nl6I2o5d3d_z7c4s_E6sA";

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

export async function fetchDocuments() {
  console.log("Fetching documents from Supabase...");
  const { data, error } = await supabase
    .from("documents")
    .select("id, title, metadata, umap_coordinates");

  if (error) {
    console.error("Error fetching from Supabase:", error);
    throw error;
  }

  console.log(`Fetched ${data.length} documents.`);
  return data;
}
