import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ variable: "--font-inter", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "ParkPulse — Parking Enforcement Intelligence for Bengaluru",
  description:
    "Turning 298,000 parking-violation records into targeted enforcement: detect hotspots, " +
    "score their traffic impact, forecast them, and deploy patrols. Gridlock Hackathon 2.0.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${inter.variable} h-full antialiased`}>
      <body className="min-h-full">{children}</body>
    </html>
  );
}
