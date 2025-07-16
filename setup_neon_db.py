#!/usr/bin/env python3
"""
Setup script for Neon PostgreSQL with vector database support
This script helps configure and test the vector database integration.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.core.database import init_database, engine, close_database
from sqlalchemy import text
from app.core.config import settings
from app.services.embedding_service import embedding_service
from app.services.vector_database import vector_database_service
from app.domain.models import Source
from app.core.logging import setup_logging, get_logger

setup_logging()
logger = get_logger()


async def test_database_connection():
    """Test database connection and basic operations"""
    print("🔍 Testing database connection...")
    
    try:
        # Initialize database first (creates tables and extensions)
        await init_database()
        print("✅ Database initialized successfully")
        
        # Test connection
        async with engine.begin() as conn:
            # Test basic query
            result = await conn.execute(text("SELECT 1"))
            print("✅ Database connection successful")
            
            # Check if pgvector extension is available
            result = await conn.execute(text("SELECT * FROM pg_extension WHERE extname = 'vector'"))
            if result.fetchone():
                print("✅ pgvector extension is available")
            else:
                print("❌ pgvector extension not found. Please install it in your Neon database.")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


async def test_embedding_service():
    """Test embedding service functionality"""
    print("\n🧠 Testing embedding service...")
    
    try:
        # Test embedding generation
        test_text = "The Earth is round"
        embedding = await embedding_service.get_embedding(test_text)
        
        if embedding and len(embedding) > 0:
            print(f"✅ Embedding service working (vector dimension: {len(embedding)})")
            return True
        else:
            print("❌ Embedding service failed to generate embeddings")
            return False
            
    except Exception as e:
        print(f"❌ Embedding service test failed: {e}")
        return False


async def test_vector_operations():
    """Test vector database operations"""
    print("\n🗄️ Testing vector database operations...")
    
    try:
        from sqlalchemy.ext.asyncio import AsyncSession
        from app.core.database import AsyncSessionLocal
        
        async with AsyncSessionLocal() as session:
            # Test storing a fact-check record
            test_sources = [
                Source(
                    title="Test Source 1",
                    url="https://example.com/test1",
                    snippet="This is a test source for verification."
                ),
                Source(
                    title="Test Source 2", 
                    url="https://example.com/test2",
                    snippet="Another test source for verification."
                )
            ]
            
            record = await vector_database_service.store_fact_check(
                session=session,
                claim="The Earth is round",
                verdict="True",
                confidence=95,
                reasoning="This is a test fact-check record for verification purposes.",
                tldr="The Earth is indeed round according to scientific evidence.",
                sources=test_sources
            )
            
            if record:
                print(f"✅ Successfully stored fact-check record (ID: {record.id})")
                
                # Test retrieving the record
                if record.id is not None:
                    retrieved = await vector_database_service.get_fact_check_by_id(
                        session, record.id
                    )
                else:
                    retrieved = None
                
                if retrieved:
                    print("✅ Successfully retrieved stored record")
                else:
                    print("❌ Failed to retrieve stored record")
                    return False
                
                # Test similarity search
                similar_results = await vector_database_service.find_similar_claims(
                    session, "Earth is spherical", max_results=3
                )
                
                if similar_results:
                    print(f"✅ Similarity search working (found {len(similar_results)} results)")
                else:
                    print("⚠️ Similarity search returned no results (this might be normal)")
                
                return True
            else:
                print("❌ Failed to store fact-check record")
                return False
                
    except Exception as e:
        print(f"❌ Vector operations test failed: {e}")
        return False


async def create_sample_data():
    """Create sample fact-check data for testing"""
    print("\n📝 Creating sample data...")
    
    try:
        from sqlalchemy.ext.asyncio import AsyncSession
        from app.core.database import AsyncSessionLocal
        
        sample_claims = [
            {
                "claim": "The Earth is round",
                "verdict": "True",
                "confidence": 95,
                "reasoning": "The Earth is an oblate spheroid, which is a type of round shape. This has been confirmed by centuries of scientific observation and modern satellite imagery.",
                "tldr": "The Earth is indeed round, specifically an oblate spheroid.",
                "sources": [
                    Source(
                        title="NASA - Earth",
                        url="https://www.nasa.gov/earth",
                        snippet="Earth is the third planet from the Sun and the only astronomical object known to harbor life."
                    )
                ]
            },
            {
                "claim": "Vaccines cause autism",
                "verdict": "False", 
                "confidence": 98,
                "reasoning": "Multiple large-scale studies have found no link between vaccines and autism. The original study that suggested this link has been thoroughly debunked and retracted.",
                "tldr": "Vaccines do not cause autism. This claim has been thoroughly debunked by scientific research.",
                "sources": [
                    Source(
                        title="CDC - Vaccine Safety",
                        url="https://www.cdc.gov/vaccinesafety/concerns/autism.html",
                        snippet="Studies have shown that there is no link between receiving vaccines and developing autism spectrum disorder."
                    )
                ]
            },
            {
                "claim": "Coffee is good for health",
                "verdict": "Unclear",
                "confidence": 75,
                "reasoning": "Coffee has both potential health benefits and risks. Moderate consumption may have some benefits, but effects vary by individual and depend on amount consumed.",
                "tldr": "Coffee's health effects are complex and depend on individual factors and consumption amount.",
                "sources": [
                    Source(
                        title="Harvard Health - Coffee and Health",
                        url="https://www.health.harvard.edu/coffee",
                        snippet="Coffee contains hundreds of bioactive compounds that may have health benefits."
                    )
                ]
            }
        ]
        
        async with AsyncSessionLocal() as session:
            stored_count = 0
            for sample in sample_claims:
                record = await vector_database_service.store_fact_check(
                    session=session,
                    claim=sample["claim"],
                    verdict=sample["verdict"],
                    confidence=sample["confidence"],
                    reasoning=sample["reasoning"],
                    tldr=sample["tldr"],
                    sources=sample["sources"]
                )
                
                if record:
                    stored_count += 1
                    print(f"  ✅ Stored: {sample['claim'][:50]}...")
                else:
                    print(f"  ❌ Failed to store: {sample['claim'][:50]}...")
            
            print(f"✅ Created {stored_count} sample records")
            return stored_count > 0
            
    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        return False


async def main():
    """Main setup function"""
    print("🚀 FactCheck AI - Neon PostgreSQL Vector Database Setup")
    print("=" * 60)
    
    # Check environment variables
    print("\n📋 Checking configuration...")
    
    required_vars = [
        "HUGGINGFACE_API_KEY",
        "SERPER_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file")
        return False
    
    # Check database configuration
    if not settings.neon_database_url and not all([
        settings.neon_database_host, 
        settings.neon_database_user, 
        settings.neon_database_password
    ]):
        print("❌ Database configuration incomplete")
        print("Please set NEON_DATABASE_URL or all individual database settings")
        return False
    
    print("✅ Configuration looks good")
    
    # Run tests
    tests = [
        ("Database Connection", test_database_connection),
        ("Embedding Service", test_embedding_service),
        ("Vector Operations", test_vector_operations),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        try:
            result = await test_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Vector database is ready to use.")
        
        # Ask if user wants to create sample data
        try:
            response = input("\nWould you like to create sample data? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                await create_sample_data()
        except KeyboardInterrupt:
            print("\nSetup cancelled by user")
        
        print("\n✅ Setup complete! You can now run your FactCheck AI application.")
        return True
    else:
        print("\n❌ Some tests failed. Please check your configuration and try again.")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with unexpected error: {e}")
        sys.exit(1) 