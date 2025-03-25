'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import SentimentChart from "@/components/dashboard/SentimentChart";
import SentimentDistributionChart from "@/components/dashboard/SentimentDistributionChart";
import SentimentShiftChart from "@/components/dashboard/SentimentShiftChart";

// Define types for our data
interface SentimentDistribution {
  VERY_POSITIVE: number;
  POSITIVE: number;
  NEUTRAL: number;
  NEGATIVE: number;
  VERY_NEGATIVE: number;
}

interface SubnetData {
  subnet: number;
  activity_level: number;
  avg_sentiment: number;
  sentiment_distribution: SentimentDistribution;
  future_related_count: number;
  future_related_percentage: number;
  technical_count: number;
  technical_percentage: number;
  timestamp: string;
  is_golden?: boolean;
  tier?: string;
  cross_references: {
    source_subnet: number;
    sentiment: number;
    insight: string;
    author: string;
    timestamp: string;
  }[];
}

// Helper functions for sentiment
const getSentimentColor = (sentiment: number): string => {
  if (sentiment > 0.6) return 'bg-green-500';
  if (sentiment > 0.2) return 'bg-green-300';
  if (sentiment > -0.2) return 'bg-gray-300';
  if (sentiment > -0.6) return 'bg-red-300';
  return 'bg-red-500';
};

const getSentimentLabel = (sentiment: number): string => {
  if (sentiment > 0.6) return 'Very Positive';
  if (sentiment > 0.2) return 'Positive';
  if (sentiment > -0.2) return 'Neutral';
  if (sentiment > -0.6) return 'Negative';
  return 'Very Negative';
};

// Style for golden subnets
const goldenSubnetStyle = "border-2 border-yellow-400 bg-yellow-50";

// Mock data for initial render - will be replaced with real data
const mockSubnetData: SubnetData[] = Array.from({ length: 5 }, (_, i) => ({
  subnet: i + 1,
  activity_level: 0,
  avg_sentiment: 0,
  sentiment_distribution: {
    VERY_POSITIVE: 0,
    POSITIVE: 0,
    NEUTRAL: 0,
    NEGATIVE: 0,
    VERY_NEGATIVE: 0
  },
  future_related_count: 0,
  future_related_percentage: 0,
  technical_count: 0,
  technical_percentage: 0,
  timestamp: new Date().toISOString(),
  is_golden: false,
  cross_references: []
}));

// Compact subnet card for grid layout
const CompactSubnetCard = ({ data }: { data: SubnetData }) => {
  const getRatingBadge = (rating: string | null) => {
    if (!rating) return null;
    
    const badgeStyle = "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium mr-2";
    
    // Handle any tier format by checking if the string contains our keywords
    const lowerRating = rating.toLowerCase();
    if (lowerRating.includes('emerald')) {
      return <span className={`${badgeStyle} bg-emerald-100 text-emerald-800`}>🔹 Emerald</span>;
    } else if (lowerRating.includes('gold')) {
      return <span className={`${badgeStyle} bg-yellow-100 text-yellow-800`}>✨ Golden</span>;
    } else if (lowerRating.includes('silver')) {
      return <span className={`${badgeStyle} bg-gray-200 text-gray-800`}>⚪ Silver</span>;
    } else if (lowerRating.includes('bronze')) {
      return <span className={`${badgeStyle} bg-amber-100 text-amber-800`}>🟤 Bronze</span>;
    } else if (lowerRating === 'standard') {
      return <span className={`${badgeStyle} bg-gray-100 text-gray-800`}>Standard</span>;
    } else {
      return <span className={`${badgeStyle} bg-blue-100 text-blue-800`}>{rating}</span>;
    }
  };

  // Format sentiment for display
  const formattedSentiment = data.avg_sentiment.toFixed(2);
  
  return (
    <Card className={`h-full ${data.is_golden ? goldenSubnetStyle : ''}`}>
      <CardHeader className="p-3 pb-0">
        <div className="flex justify-between items-center">
          <CardTitle className="text-lg">Subnet {data.subnet}</CardTitle>
          {data.is_golden && <span className="text-sm">🌟</span>}
        </div>
        {data.tier && (
          <div className="mt-1">
            {getRatingBadge(data.tier)}
          </div>
        )}
      </CardHeader>
      <CardContent className="p-3">
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-sm font-medium">Sentiment</span>
            <Badge className={getSentimentColor(data.avg_sentiment)}>
              {formattedSentiment}
            </Badge>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-sm font-medium">Activity</span>
            <span className="text-sm">{data.activity_level > 0 ? 'Active' : 'Inactive'}</span>
          </div>
          <div className="h-2 w-full bg-gray-200 rounded-full overflow-hidden">
            <div
              className={`h-full ${getSentimentColor(data.avg_sentiment)}`}
              style={{ width: `${Math.abs(data.avg_sentiment) * 100}%` }}
            ></div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Update the Grid component to use production tiers
const DashboardGrid = ({ data }: { data: SubnetData[] }) => {
  // Sort subnets by number
  const sortedData = [...data].sort((a, b) => a.subnet - b.subnet);
  
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 p-4">
      {sortedData.map((subnet) => (
        <CompactSubnetCard key={subnet.subnet} data={subnet} />
      ))}
    </div>
  );
};

export function Dashboard() {
  const [subnets, setSubnets] = useState<SubnetData[]>(mockSubnetData);
  const [loading, setLoading] = useState(true);
  const [historyData, setHistoryData] = useState<any[]>([]);
  const [activeTab, setActiveTab] = useState('overview');

  // Fetch sentiment data
  useEffect(() => {
    const fetchData = async () => {
      try {
        // In production, this would fetch from your API
        // For now, we're simulating data
        const response = await fetch('/data/all_subnets_latest.json');
        if (response.ok) {
          const data = await response.json();
          // Process data array format (our JSON is an array of subnet objects)
          const dataArray = data.map((subnetData: any) => {
            // Convert sentiment_distribution from new format to the format dashboard expects
            const sentDistribution = subnetData.sentiment_distribution || {};
            const formattedDistribution = {
              VERY_POSITIVE: sentDistribution.positive || 0,
              POSITIVE: sentDistribution.positive || 0,
              NEUTRAL: sentDistribution.neutral || 0,
              NEGATIVE: sentDistribution.negative || 0,
              VERY_NEGATIVE: sentDistribution.negative || 0
            };
            
            // Create a properly formatted subnet entry
            return {
              subnet: parseInt(subnetData.subnet),
              activity_level: subnetData.message_count || 0,
              avg_sentiment: subnetData.average_sentiment || 0,
              sentiment_distribution: formattedDistribution,
              future_related_count: 0, // Placeholder, we don't have this in new format
              future_related_percentage: 0,
              technical_count: 0, // Placeholder, we don't have this in new format
              technical_percentage: 0,
              timestamp: subnetData.last_updated || new Date().toISOString(),
              is_golden: subnetData.is_golden || false,
              tier: subnetData.tier || null,
              cross_references: []
            };
          });
          
          // Sort by sentiment - most positive first
          const sortedData = dataArray.sort((a: SubnetData, b: SubnetData) => b.avg_sentiment - a.avg_sentiment);
          setSubnets(sortedData);
        } else {
          // If file doesn't exist yet, simulate data
          const simulatedData = Array.from({ length: 80 }, (_, i) => ({
            subnet: i + 1,
            activity_level: Math.floor(Math.random() * 500),
            avg_sentiment: (Math.random() * 2 - 1),
            sentiment_distribution: {
              VERY_POSITIVE: Math.random() * 20,
              POSITIVE: Math.random() * 30,
              NEUTRAL: Math.random() * 40,
              NEGATIVE: Math.random() * 20,
              VERY_NEGATIVE: Math.random() * 10
            },
            future_related_count: Math.floor(Math.random() * 100),
            future_related_percentage: Math.random() * 30,
            technical_count: Math.floor(Math.random() * 300),
            technical_percentage: Math.random() * 70,
            timestamp: new Date().toISOString(),
            is_golden: false,
            cross_references: []
          }));
          
          const sortedSimulatedData = [...simulatedData].sort((a: SubnetData, b: SubnetData) => b.avg_sentiment - a.avg_sentiment);
          setSubnets(sortedSimulatedData);
        }
        
        // Also load some fake history data for charts
        const historyData = Array.from({ length: 10 }, (_, i) => ({
          date: new Date(Date.now() - (9 - i) * 12 * 60 * 60 * 1000).toISOString(),
          sentiment: Math.random() * 2 - 1
        }));
        setHistoryData(historyData);
        
      } catch (error) {
        console.error("Error fetching data:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Get top 3 most positive and most negative subnets
  const topPositive = [...subnets].sort((a: SubnetData, b: SubnetData) => b.avg_sentiment - a.avg_sentiment).slice(0, 3);
  const topNegative = [...subnets].sort((a: SubnetData, b: SubnetData) => a.avg_sentiment - b.avg_sentiment).slice(0, 3);

  return (
    <div className="space-y-6">
      <Tabs defaultValue="overview" className="space-y-4" onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="subnets">All Subnets</TabsTrigger>
          <TabsTrigger value="trends">Sentiment Trends</TabsTrigger>
          <TabsTrigger value="future">Future Anticipation</TabsTrigger>
        </TabsList>
        
        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle>Overall Sentiment</CardTitle>
                <CardDescription>Average across all subnets</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {subnets.length > 0 && (
                    subnets.reduce((sum, subnet) => sum + subnet.avg_sentiment, 0) / subnets.length
                  ).toFixed(2)}
                </div>
                <div className="text-xs text-muted-foreground">
                  {subnets.length > 0 && getSentimentLabel(
                    subnets.reduce((sum, subnet) => sum + subnet.avg_sentiment, 0) / subnets.length
                  )}
                </div>
                <SentimentChart data={historyData} />
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle>Top Positive Subnets</CardTitle>
                <CardDescription>Highest sentiment scores</CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {topPositive.map((subnet) => (
                    <li key={`positive-${subnet.subnet}`} className={`flex items-center justify-between ${subnet.is_golden ? 'p-1 ' + goldenSubnetStyle : ''}`}>
                      <div className="flex items-center gap-2">
                        <span>Subnet {subnet.subnet}</span>
                        {subnet.is_golden && (
                          <Badge className="bg-yellow-400 text-black border-0 text-xs">
                            🌟
                          </Badge>
                        )}
                      </div>
                      <Badge className={getSentimentColor(subnet.avg_sentiment)}>
                        {subnet.avg_sentiment.toFixed(2)}
                      </Badge>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle>Top Negative Subnets</CardTitle>
                <CardDescription>Lowest sentiment scores</CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {topNegative.map((subnet) => (
                    <li key={`negative-${subnet.subnet}`} className="flex items-center justify-between">
                      <span>Subnet {subnet.subnet}</span>
                      <Badge className={getSentimentColor(subnet.avg_sentiment)}>
                        {subnet.avg_sentiment.toFixed(2)}
                      </Badge>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          </div>
          
          <Card>
            <CardHeader>
              <CardTitle>Sentiment Distribution</CardTitle>
              <CardDescription>Across all analyzed messages</CardDescription>
            </CardHeader>
            <CardContent>
              {subnets.length > 0 && (
                <SentimentDistributionChart
                  data={Object.entries(
                    subnets.reduce((acc, subnet) => {
                      Object.entries(subnet.sentiment_distribution).forEach(([key, value]) => {
                        acc[key] = (acc[key] || 0) + value;
                      });
                      return acc;
                    }, {} as Record<string, number>)
                  ).reduce((obj, [key, value]) => {
                    obj[key] = value;
                    return obj;
                  }, {} as Record<string, number>)}
                />
              )}
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle>Recent Sentiment Shifts</CardTitle>
              <CardDescription>Changes in the last 5 days</CardDescription>
            </CardHeader>
            <CardContent>
              <SentimentShiftChart data={historyData} />
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* All Subnets Tab */}
        <TabsContent value="subnets">
          <DashboardGrid data={subnets} />
        </TabsContent>
        
        {/* Trends Tab */}
        <TabsContent value="trends">
          <Card>
            <CardHeader>
              <CardTitle>Sentiment Trends Over Time</CardTitle>
              <CardDescription>
                Tracking sentiment changes across subnets
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-6">
              <div className="text-center mb-6">
                <p>Simulated data for demonstration purposes.</p>
                <p className="text-muted-foreground text-sm mt-1">
                  In production, this would show real historical trends.
                </p>
              </div>
              <div className="h-96">
                <SentimentShiftChart data={historyData} />
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* Future Anticipation Tab */}
        <TabsContent value="future">
          <Card>
            <CardHeader>
              <CardTitle>Future Anticipation Analysis</CardTitle>
              <CardDescription>
                Messages expressing excitement about upcoming features
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-8">
                <div className="space-y-4">
                  <h3 className="text-lg font-medium">Top Subnets Discussing Future Features</h3>
                  <ul className="space-y-3">
                    {[...subnets]
                      .sort((a: SubnetData, b: SubnetData) => b.future_related_percentage - a.future_related_percentage)
                      .slice(0, 5)
                      .map((subnet) => (
                        <li key={`future-${subnet.subnet}`} className="flex items-center justify-between">
                          <div>
                            <span className="font-medium">Subnet {subnet.subnet}</span>
                            <p className="text-sm text-muted-foreground">
                              {subnet.future_related_count} messages ({subnet.future_related_percentage.toFixed(1)}%)
                            </p>
                          </div>
                          <Badge variant="outline">
                            {getSentimentLabel(subnet.avg_sentiment)}
                          </Badge>
                        </li>
                      ))}
                  </ul>
                </div>
                
                <div className="rounded-md border p-4 bg-muted/50">
                  <h3 className="text-lg font-medium mb-2">OpenRouter AI Analysis</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Deep analysis uses OpenRouter with Qwen 32B to detect nuanced anticipation expressions in messages.
                  </p>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm font-medium">Analysis powered by</span>
                    <Badge variant="outline" className="bg-blue-50">Qwen 32B</Badge>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <h3 className="text-lg font-medium">Top Future-Related Topics</h3>
                  <div className="grid gap-2 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
                    {Array.from({ length: 6 }).map((_, i) => (
                      <Card key={`topic-${i}`} className="hover:bg-muted/50 transition-colors">
                        <CardHeader className="p-4 pb-2">
                          <CardTitle className="text-base">
                            {["L2 Integration", "Token Rewards", "Multi-Sig Support", 
                              "Cross-Chain Assets", "Subnet Economics", "Mobile App"][i]}
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="p-4 pt-0">
                          <div className="text-sm text-muted-foreground">
                            Mentioned in {Math.floor(Math.random() * 40) + 10} messages
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default Dashboard;
