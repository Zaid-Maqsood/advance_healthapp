import aiohttp
import json
import re
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DailyMedTool:
    """Tool for querying medication information from DailyMed API"""
    
    def __init__(self):
        self.base_url = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search_drug(self, drug_name: str) -> Dict[str, Any]:
        """Search for a drug by name in DailyMed database"""
        try:
            search_url = f"{self.base_url}/drugnames.json"
            params = {"name": drug_name}
            
            async with self.session.get(search_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "drugs": data.get("data", []),
                        "query": drug_name
                    }
                else:
                    return {
                        "success": False,
                        "error": f"API returned status {response.status}",
                        "query": drug_name
                    }
        except Exception as e:
            logger.error(f"Error searching drug {drug_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": drug_name
            }
    
    async def get_drug_details(self, set_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific drug"""
        try:
            drug_url = f"{self.base_url}/spls/{set_id}.json"
            
            async with self.session.get(drug_url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "drug_info": self._parse_drug_info(data),
                        "set_id": set_id
                    }
                else:
                    return {
                        "success": False,
                        "error": f"API returned status {response.status}",
                        "set_id": set_id
                    }
        except Exception as e:
            logger.error(f"Error getting drug details for {set_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "set_id": set_id
            }
    
    def _parse_drug_info(self, data: Dict) -> Dict[str, Any]:
        """Parse drug information from DailyMed API response"""
        try:
            # Extract the main drug data
            drug_data = data.get("data", {})
            
            # Parse structured product labeling (SPL) data
            spl_data = drug_data.get("spl", {})
            
            return {
                "name": drug_data.get("name", "Unknown"),
                "set_id": drug_data.get("set_id", ""),
                "active_ingredients": self._extract_active_ingredients(spl_data),
                "indications": self._extract_indications(spl_data),
                "contraindications": self._extract_contraindications(spl_data),
                "warnings": self._extract_warnings(spl_data),
                "adverse_reactions": self._extract_adverse_reactions(spl_data),
                "dosage_administration": self._extract_dosage_administration(spl_data),
                "storage_handling": self._extract_storage_handling(spl_data),
                "manufacturer": drug_data.get("manufacturer", "Unknown"),
                "last_updated": drug_data.get("last_updated", "Unknown")
            }
        except Exception as e:
            logger.error(f"Error parsing drug info: {str(e)}")
            return {"error": f"Failed to parse drug info: {str(e)}"}
    
    def _extract_active_ingredients(self, spl_data: Dict) -> List[str]:
        """Extract active ingredients from SPL data"""
        try:
            ingredients = []
            # Look for active ingredients in various possible locations
            if "active_ingredient" in spl_data:
                if isinstance(spl_data["active_ingredient"], list):
                    ingredients = [ing.get("name", "") for ing in spl_data["active_ingredient"]]
                else:
                    ingredients = [spl_data["active_ingredient"].get("name", "")]
            return [ing for ing in ingredients if ing]
        except:
            return []
    
    def _extract_indications(self, spl_data: Dict) -> List[str]:
        """Extract indications from SPL data"""
        try:
            indications = []
            # Look for indications in various possible locations
            if "indications_and_usage" in spl_data:
                content = spl_data["indications_and_usage"]
                if isinstance(content, str):
                    indications = [content]
                elif isinstance(content, list):
                    indications = [item.get("text", "") for item in content if isinstance(item, dict)]
            return [ind for ind in indications if ind]
        except:
            return []
    
    def _extract_contraindications(self, spl_data: Dict) -> List[str]:
        """Extract contraindications from SPL data"""
        try:
            contraindications = []
            if "contraindications" in spl_data:
                content = spl_data["contraindications"]
                if isinstance(content, str):
                    contraindications = [content]
                elif isinstance(content, list):
                    contraindications = [item.get("text", "") for item in content if isinstance(item, dict)]
            return [cont for cont in contraindications if cont]
        except:
            return []
    
    def _extract_warnings(self, spl_data: Dict) -> List[str]:
        """Extract warnings from SPL data"""
        try:
            warnings = []
            if "warnings_and_cautions" in spl_data:
                content = spl_data["warnings_and_cautions"]
                if isinstance(content, str):
                    warnings = [content]
                elif isinstance(content, list):
                    warnings = [item.get("text", "") for item in content if isinstance(item, dict)]
            return [warn for warn in warnings if warn]
        except:
            return []
    
    def _extract_adverse_reactions(self, spl_data: Dict) -> List[str]:
        """Extract adverse reactions from SPL data"""
        try:
            reactions = []
            if "adverse_reactions" in spl_data:
                content = spl_data["adverse_reactions"]
                if isinstance(content, str):
                    reactions = [content]
                elif isinstance(content, list):
                    reactions = [item.get("text", "") for item in content if isinstance(item, dict)]
            return [reaction for reaction in reactions if reaction]
        except:
            return []
    
    def _extract_dosage_administration(self, spl_data: Dict) -> List[str]:
        """Extract dosage and administration from SPL data"""
        try:
            dosages = []
            if "dosage_and_administration" in spl_data:
                content = spl_data["dosage_and_administration"]
                if isinstance(content, str):
                    dosages = [content]
                elif isinstance(content, list):
                    dosages = [item.get("text", "") for item in content if isinstance(item, dict)]
            return [dose for dose in dosages if dose]
        except:
            return []
    
    def _extract_storage_handling(self, spl_data: Dict) -> List[str]:
        """Extract storage and handling from SPL data"""
        try:
            storage = []
            if "storage_and_handling" in spl_data:
                content = spl_data["storage_and_handling"]
                if isinstance(content, str):
                    storage = [content]
                elif isinstance(content, list):
                    storage = [item.get("text", "") for item in content if isinstance(item, dict)]
            return [stor for stor in storage if stor]
        except:
            return []

class MedicationQueryTool:
    """Main tool for handling medication queries with DailyMed integration"""
    
    def __init__(self):
        self.name = "medication_query_tool"
        self.description = "Query medication information from DailyMed database"
        self.category = "external_api"
        self.parameters = {
            "drug_name": {"type": "string", "description": "Name of the medication to query"},
            "query_type": {"type": "string", "description": "Type of query: 'search', 'details'"}
        }
    
    async def query_medication(self, drug_name: str, query_type: str = "search") -> Dict[str, Any]:
        """Main method to query medication information"""
        async with DailyMedTool() as dailymed:
            if query_type == "search":
                return await dailymed.search_drug(drug_name)
            elif query_type == "details":
                # First search to get drug ID, then get details
                search_result = await dailymed.search_drug(drug_name)
                if search_result["success"] and search_result["drugs"]:
                    # Get the first matching drug
                    drug = search_result["drugs"][0]
                    set_id = drug.get("set_id")
                    if set_id:
                        return await dailymed.get_drug_details(set_id)
                return search_result
            else:
                return {
                    "success": False,
                    "error": f"Unknown query type: {query_type}"
                }
    
    def format_medication_response(self, drug_info: Dict[str, Any], query: str) -> str:
        """Format medication information for user response"""
        if not drug_info.get("success"):
            return f"Sorry, I couldn't find information about this medication in the DailyMed database. Error: {drug_info.get('error', 'Unknown error')}"
        
        drug_data = drug_info.get("drug_info", {})
        
        # Start with disclaimer
        response = "According to DailyMed, which is maintained by the U.S. National Library of Medicine (NLM), which is part of the U.S. National Institutes of Health (NIH):\n\n"
        
        # Basic information
        response += f"**{drug_data.get('name', 'Unknown Medication')}**\n\n"
        
        # Active ingredients
        active_ingredients = drug_data.get('active_ingredients', [])
        if active_ingredients:
            response += f"**Active Ingredients:** {', '.join(active_ingredients)}\n\n"
        
        # Indications
        indications = drug_data.get('indications', [])
        if indications:
            response += f"**Indications:** {indications[0][:500]}{'...' if len(indications[0]) > 500 else ''}\n\n"
        
        # Warnings
        warnings = drug_data.get('warnings', [])
        if warnings:
            response += f"**Important Warnings:** {warnings[0][:500]}{'...' if len(warnings[0]) > 500 else ''}\n\n"
        
        # Contraindications
        contraindications = drug_data.get('contraindications', [])
        if contraindications:
            response += f"**Contraindications:** {contraindications[0][:500]}{'...' if len(contraindications[0]) > 500 else ''}\n\n"
        
        # Adverse reactions
        adverse_reactions = drug_data.get('adverse_reactions', [])
        if adverse_reactions:
            response += f"**Common Side Effects:** {adverse_reactions[0][:500]}{'...' if len(adverse_reactions[0]) > 500 else ''}\n\n"
        
        # Dosage information
        dosage = drug_data.get('dosage_administration', [])
        if dosage:
            response += f"**Dosage Information:** {dosage[0][:500]}{'...' if len(dosage[0]) > 500 else ''}\n\n"
        
        # Storage information
        storage = drug_data.get('storage_handling', [])
        if storage:
            response += f"**Storage Instructions:** {storage[0][:300]}{'...' if len(storage[0]) > 300 else ''}\n\n"
        
        # Manufacturer and last updated
        manufacturer = drug_data.get('manufacturer', 'Unknown')
        last_updated = drug_data.get('last_updated', 'Unknown')
        response += f"**Manufacturer:** {manufacturer}\n"
        response += f"**Last Updated:** {last_updated}\n\n"
        
        # Safety disclaimers
        response += "⚠️ **IMPORTANT:** This information is for educational purposes only. Always consult your doctor before taking any medication.\n\n"
        
        # Additional disclaimer based on query type
        if any(word in query.lower() for word in ['good', 'best', 'recommend', 'should i take', 'which medicine']):
            response += "**Please discuss with your doctor:** This information should not be used to make decisions about which medication to take. Your doctor will determine the best treatment for your specific condition.\n\n"
        
        if any(word in query.lower() for word in ['disease', 'condition', 'treatment', 'cure']):
            response += "**Please discuss with your doctor:** This medication information should not be used to self-treat any condition. Always consult your healthcare provider for proper diagnosis and treatment.\n\n"
        
        return response
