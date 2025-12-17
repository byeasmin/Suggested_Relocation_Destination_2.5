import os
import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from fastapi.middleware.cors import CORSMiddleware
import math
import asyncio

app = FastAPI(
    title="NASA Relocation Advisor",
    description="Find safer relocation destinations using NASA environmental data",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AQI dataset
try:
    df_aqi = pd.read_csv("AQI and Lat Long of Countries.csv")
    print(f"Loaded AQI dataset with {len(df_aqi)} records")
except FileNotFoundError:
    df_aqi = None
    print("Warning: AQI dataset not found")

# Models
class RelocationRequest(BaseModel):
    current_latitude: float = Field(..., ge=-90, le=90)
    current_longitude: float = Field(..., ge=-180, le=180)
    search_radius_km: int = Field(500, ge=50, le=2000)
    max_recommendations: int = Field(5, ge=1, le=10)
    priorities: List[str] = Field(["air_quality", "solar_potential"])

class LocationScore(BaseModel):
    country: str
    city: str
    latitude: float
    longitude: float
    overall_score: float
    air_quality_score: float
    solar_score: float
    temperature_score: float
    safety_status: str
    improvement: float
    distance_km: float
    nasa_data_quality: str

class RelocationResponse(BaseModel):
    current_location_analysis: Dict
    recommended_locations: List[LocationScore]
    relocation_analysis: str
    top_recommendation: Dict
    search_parameters: Dict

# NASA POWER API Service
class NASAPowerService:
    @staticmethod
    def adjust_coordinates(lat: float, lon: float):
        """Adjust coordinates to valid ranges"""
        while lat > 90 or lat < -90:
            if lat > 90:
                lat = 180 - lat
            elif lat < -90:
                lat = -180 - lat
        lon = ((lon + 180) % 360) - 180
        return lat, lon

    @staticmethod
    async def get_nasa_environmental_data(lat: float, lon: float) -> Dict:
        """Get comprehensive environmental data from NASA POWER API"""
        try:
            lat_adj, lon_adj = NASAPowerService.adjust_coordinates(lat, lon)
            
            # NASA POWER API call for multiple parameters
            power_api_url = (
                'https://power.larc.nasa.gov/api/temporal/daily/point'
                '?parameters=T2M,RH2M,WS2M,ALLSKY_SFC_SW_DWN,PRECTOT'
                '&community=RE'
                f'&longitude={lon_adj}'
                f'&latitude={lat_adj}'
                '&start=20240101&end=20240101'
                '&format=JSON'
            )

            async with httpx.AsyncClient() as client:
                response = await client.get(power_api_url, timeout=20.0)
                response.raise_for_status()
                nasa_data = response.json()

            parameters = nasa_data.get("properties", {}).get("parameter", {})
            
            if not parameters:
                return {"error": "No NASA data available for this location"}
            
            # Extract first available values
            def get_first_valid_value(param_dict):
                if not param_dict:
                    return None
                for value in param_dict.values():
                    if value != -999 and value is not None:  # -999 is NASA's no-data value
                        return value
                return None
            
            temperature = get_first_valid_value(parameters.get("T2M", {}))
            humidity = get_first_valid_value(parameters.get("RH2M", {}))
            wind_speed = get_first_valid_value(parameters.get("WS2M", {}))
            solar_radiation = get_first_valid_value(parameters.get("ALLSKY_SFC_SW_DWN", {}))
            precipitation = get_first_valid_value(parameters.get("PRECTOT", {}))
            
            # Calculate scores
            air_quality_score = NASAPowerService.calculate_air_quality_score(temperature, humidity, wind_speed)
            solar_score = NASAPowerService.calculate_solar_score(solar_radiation)
            temperature_score = NASAPowerService.calculate_temperature_score(temperature)
            
            # Overall environmental score
            overall_score = NASAPowerService.calculate_overall_score(
                air_quality_score, solar_score, temperature_score
            )
            
            return {
                "latitude": lat,
                "longitude": lon,
                "temperature": temperature,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "solar_radiation": solar_radiation,
                "precipitation": precipitation,
                "air_quality_score": air_quality_score,
                "solar_score": solar_score,
                "temperature_score": temperature_score,
                "overall_score": overall_score,
                "safety_status": NASAPowerService.get_safety_status(overall_score),
                "data_quality": "good" if temperature is not None else "poor"
            }
            
        except Exception as e:
            return {"error": f"NASA API error: {str(e)}"}

    @staticmethod
    def calculate_air_quality_score(temperature: float, humidity: float, wind_speed: float) -> float:
        """Calculate air quality score based on environmental factors"""
        score = 70  # Base score
        
        # Temperature effect (extreme temps can worsen air quality)
        if temperature:
            if 18 <= temperature <= 26:
                score += 20  # Ideal temperature
            elif temperature > 35 or temperature < 0:
                score -= 20  # Extreme temperatures
            elif temperature > 30 or temperature < 5:
                score -= 10  # Uncomfortable temperatures
        
        # Humidity effect
        if humidity:
            if 30 <= humidity <= 60:
                score += 10  # Comfortable humidity
            elif humidity > 80:
                score -= 15  # High humidity can trap pollutants
            elif humidity < 20:
                score -= 10  # Low humidity can cause respiratory issues
        
        # Wind speed effect
        if wind_speed:
            if 1 <= wind_speed <= 5:
                score += 10  # Good ventilation
            elif wind_speed > 10:
                score -= 10  # Too windy
            elif wind_speed < 1:
                score -= 5   # Poor air circulation
        
        return max(0, min(100, score))

    @staticmethod
    def calculate_solar_score(solar_radiation: float) -> float:
        """Calculate solar potential score"""
        if not solar_radiation or solar_radiation <= 0:
            return 50
        
        # Normalize to 0-100 scale (typical range 0-8 kWh/m²/day)
        if solar_radiation >= 6:
            return 95  # Excellent
        elif solar_radiation >= 4:
            return 80  # Very good
        elif solar_radiation >= 2:
            return 65  # Good
        elif solar_radiation >= 1:
            return 50  # Moderate
        else:
            return 30  # Poor

    @staticmethod
    def calculate_temperature_score(temperature: float) -> float:
        """Calculate temperature comfort score"""
        if not temperature:
            return 50
        
        if 18 <= temperature <= 24:
            return 95  # Perfect
        elif 15 <= temperature <= 27:
            return 80  # Comfortable
        elif 10 <= temperature <= 32:
            return 65  # Acceptable
        elif 5 <= temperature <= 35:
            return 50  # Tolerable
        else:
            return 30  # Uncomfortable

    @staticmethod
    def calculate_overall_score(air_quality: float, solar: float, temperature: float) -> float:
        """Calculate weighted overall environmental score"""
        weights = {"air_quality": 0.5, "solar": 0.3, "temperature": 0.2}
        return (
            air_quality * weights["air_quality"] +
            solar * weights["solar"] +
            temperature * weights["temperature"]
        )

    @staticmethod
    def get_safety_status(score: float) -> str:
        """Get safety status based on overall score"""
        if score >= 70:
            return "VERY SAFE ✅"
        elif score >= 65:
            return "SAFE 👍"
        elif score >= 50:
            return "MODERATE ⚠️"
        else:
            return "UNSAFE ❌"

# Location Generator
class LocationGenerator:
    @staticmethod
    def generate_candidate_locations(center_lat: float, center_lon: float, radius_km: int) -> List[Dict]:
        """Generate candidate locations within radius"""
        if df_aqi is None:
            return []
        
        candidate_locations = []
        
        for _, row in df_aqi.iterrows():
            try:
                city_lat = float(row["lat"])
                city_lon = float(row["lng"])
                distance = LocationGenerator.calculate_distance(center_lat, center_lon, city_lat, city_lon)
                
                if distance <= radius_km:
                    candidate_locations.append({
                        "country": row["Country"],
                        "city": row["City"],
                        "latitude": city_lat,
                        "longitude": city_lon,
                        "aqi_value": int(row["AQI Value"]),
                        "aqi_category": row["AQI Category"],
                        "distance_km": distance
                    })
            except (ValueError, KeyError):
                continue
        
        return candidate_locations

    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between coordinates in km"""
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# Relocation Engine
class RelocationEngine:
    @staticmethod
    async def analyze_location_safety(lat: float, lon: float) -> Dict:
        """Analyze safety of a single location using NASA data"""
        nasa_data = await NASAPowerService.get_nasa_environmental_data(lat, lon)
        
        if "error" in nasa_data:
            return {
                "latitude": lat,
                "longitude": lon,
                "overall_score": 0,
                "air_quality_score": 0,
                "solar_score": 0,
                "temperature_score": 0,
                "safety_status": "DATA UNAVAILABLE",
                "nasa_data_quality": "poor"
            }
        
        return nasa_data

    @staticmethod
    def calculate_improvement(current_score: float, candidate_score: float) -> float:
        """Calculate improvement percentage"""
        if current_score <= 0:
            return candidate_score
        return max(0, candidate_score - current_score)

# AI Analysis Service
class AIAnalysisService:
    @staticmethod
    async def generate_relocation_analysis(
        current_location: Dict,
        recommended_locations: List[Dict],
        priorities: List[str]
    ) -> str:
        """Generate AI analysis for relocation recommendations"""
        
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            return "AI analysis unavailable - GEMINI_API_KEY not set"
        
        prompt = f"""
        RELOCATION RECOMMENDATION ANALYSIS
        
        CURRENT LOCATION ANALYSIS:
        - Overall Safety Score: {current_location.get('overall_score', 0):.1f}/100
        - Safety Status: {current_location.get('safety_status', 'Unknown')}
        - Air Quality Score: {current_location.get('air_quality_score', 0):.1f}/100
        - Solar Potential: {current_location.get('solar_score', 0):.1f}/100
        - Temperature Comfort: {current_location.get('temperature_score', 0):.1f}/100
        
        USER PRIORITIES: {', '.join(priorities)}
        
        TOP RECOMMENDED LOCATIONS:
        """
        
        for i, location in enumerate(recommended_locations[:3], 1):
            prompt += f"""
        Location {i}: {location['city']}, {location['country']}
        - Overall Score: {location['overall_score']:.1f}/100 ({location['safety_status']})
        - Air Quality: {location['air_quality_score']:.1f}/100
        - Solar Potential: {location['solar_score']:.1f}/100
        - Temperature: {location['temperature_score']:.1f}/100
        - Improvement: +{location['improvement']:.1f} points
        - Distance: {location['distance_km']:.1f} km
        """
        
        prompt += """
        Provide a comprehensive relocation analysis including:
        1. Comparison of current vs recommended locations
        2. Key environmental improvements expected
        3. Health and sustainability benefits
        4. Practical relocation considerations
        5. Final recommendation
        
        Focus on data-driven insights and practical advice.
        """
        
        try:
            gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
            headers = {"Content-Type": "application/json"}
            
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{gemini_url}?key={gemini_api_key}", headers=headers, json=payload, timeout=30.0)
                response.raise_for_status()
                data = response.json()
            
            return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No analysis generated.")
        
        except Exception as e:
            return f"AI analysis failed: {str(e)}"

# MAIN RELOCATION ENDPOINT
@app.post("/suggest_relocation", response_model=RelocationResponse)
async def suggest_relocation(request: RelocationRequest):
    """
    Find safer relocation destinations using NASA environmental data
    """
    try:
        # Analyze current location
        current_analysis = await RelocationEngine.analyze_location_safety(
            request.current_latitude, request.current_longitude
        )
        
        # Generate candidate locations
        candidate_locations = LocationGenerator.generate_candidate_locations(
            request.current_latitude, request.current_longitude, request.search_radius_km
        )
        
        # Analyze all candidate locations concurrently
        analysis_tasks = []
        for location in candidate_locations:
            task = RelocationEngine.analyze_location_safety(
                location["latitude"], location["longitude"]
            )
            analysis_tasks.append(task)
        
        candidate_analyses = await asyncio.gather(*analysis_tasks)
        
        # Combine location data with NASA analysis
        recommended_locations = []
        for i, analysis in enumerate(candidate_analyses):
            if i < len(candidate_locations):
                location_data = candidate_locations[i]
                improvement = RelocationEngine.calculate_improvement(
                    current_analysis["overall_score"], analysis["overall_score"]
                )
                
                # Only recommend if significantly better
                if improvement > 10:  # At least 10 points improvement
                    recommended_locations.append({
                        "country": location_data["country"],
                        "city": location_data["city"],
                        "latitude": analysis["latitude"],
                        "longitude": analysis["longitude"],
                        "overall_score": round(analysis["overall_score"], 2),
                        "air_quality_score": round(analysis["air_quality_score"], 2),
                        "solar_score": round(analysis["solar_score"], 2),
                        "temperature_score": round(analysis["temperature_score"], 2),
                        "safety_status": analysis["safety_status"],
                        "improvement": round(improvement, 2),
                        "distance_km": round(location_data["distance_km"], 2),
                        "nasa_data_quality": analysis.get("nasa_data_quality", "unknown")
                    })
        
        # Sort by improvement (descending)
        recommended_locations.sort(key=lambda x: x["improvement"], reverse=True)
        recommended_locations = recommended_locations[:request.max_recommendations]
        
        # Generate AI analysis
        relocation_analysis = await AIAnalysisService.generate_relocation_analysis(
            current_analysis, recommended_locations, request.priorities
        )
        
        # Prepare response
        top_recommendation = recommended_locations[0] if recommended_locations else {}
        
        return RelocationResponse(
            current_location_analysis={
                "latitude": current_analysis["latitude"],
                "longitude": current_analysis["longitude"],
                "overall_score": round(current_analysis["overall_score"], 2),
                "safety_status": current_analysis["safety_status"],
                "air_quality_score": round(current_analysis["air_quality_score"], 2),
                "solar_score": round(current_analysis["solar_score"], 2),
                "temperature_score": round(current_analysis["temperature_score"], 2)
            },
            recommended_locations=recommended_locations,
            relocation_analysis=relocation_analysis,
            top_recommendation=top_recommendation,
            search_parameters={
                "search_radius_km": request.search_radius_km,
                "locations_analyzed": len(candidate_locations),
                "recommendations_found": len(recommended_locations),
                "priorities": request.priorities
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Relocation analysis failed: {str(e)}")

# Quick relocation suggestion
@app.get("/quick_relocation")
async def quick_relocation(
    lat: float = Query(40.7128, description="Current latitude"),
    lon: float = Query(-74.0060, description="Current longitude"),
    radius_km: int = Query(500, description="Search radius in km")
):
    """Quick relocation suggestion"""
    request = RelocationRequest(
        current_latitude=lat,
        current_longitude=lon,
        search_radius_km=radius_km,
        max_recommendations=3,
        priorities=["air_quality", "solar_potential"]
    )
    return await suggest_relocation(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)