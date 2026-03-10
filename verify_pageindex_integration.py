#!/usr/bin/env python3
"""
Verification script for PageIndex RAG integration.
Tests all components are properly wired together.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def verify_imports():
    """Verify all necessary imports work"""
    print("📦 Checking imports...")
    try:
        from src.rag.pageindex import query_for_evaluation, process_pageindex_query
        print("  ✅ pageindex module imports OK")
        
        from src.evaluation.ragas_evaluator import RAGASEvaluator, evaluate_pageindex_rag, get_rag_name
        print("  ✅ ragas_evaluator imports OK")
        
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


def verify_evaluator_instantiation():
    """Verify evaluator can be instantiated with pageindex type"""
    print("\n🔧 Testing RAGASEvaluator instantiation...")
    try:
        from src.evaluation.ragas_evaluator import RAGASEvaluator
        
        evaluator = RAGASEvaluator(rag_type="pageindex")
        print(f"  ✅ RAGASEvaluator instantiated: {evaluator.rag_name}")
        
        # Check query function is properly assigned
        if evaluator.query_function is not None:
            print(f"  ✅ Query function assigned: {evaluator.query_function.__name__}")
            return True
        else:
            print("  ❌ Query function not assigned")
            return False
    except Exception as e:
        print(f"  ❌ Instantiation failed: {e}")
        return False


def verify_all_rag_types():
    """Verify all 5 RAG types can be instantiated"""
    print("\n🎯 Testing all 5 RAG types...")
    from src.evaluation.ragas_evaluator import RAGASEvaluator, get_rag_name
    
    rag_types = ["simple", "hybrid", "hyde", "rewriter", "pageindex"]
    results = {}
    
    for rag_type in rag_types:
        try:
            evaluator = RAGASEvaluator(rag_type=rag_type)
            rag_name = get_rag_name(rag_type)
            results[rag_type] = True
            print(f"  ✅ {rag_type:12} → {rag_name}")
        except Exception as e:
            results[rag_type] = False
            print(f"  ❌ {rag_type:12} → {str(e)[:50]}")
    
    return all(results.values())


def verify_cli_dispatch():
    """Verify evaluator main() has pageindex case"""
    print("\n💻 Checking CLI dispatch...")
    try:
        with open(project_root / "src" / "evaluation" / "ragas_evaluator.py", "r") as f:
            content = f.read()
        
        checks = {
            "pageindex import": 'from src.rag.pageindex import' in content,
            "pageindex elif": 'elif rag_type == "pageindex":' in content,
            "pageindex in rag_types": '"pageindex"' in content and 'rag_types = [' in content,
            "evaluate_pageindex_rag exists": 'def evaluate_pageindex_rag(' in content,
            "pageindex help text": '- pageindex: PageIndex RAG' in content or '- pageindex:' in content,
        }
        
        for check_name, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}")
        
        return all(checks.values())
    except Exception as e:
        print(f"  ❌ Failed to read evaluator: {e}")
        return False


def verify_dependencies():
    """Verify required packages are installed"""
    print("\n📚 Checking dependencies...")
    required_packages = {
        "pageindex": "pageindex",
        "langchain": "langchain",
        "langchain_community": "langchain_community",
        "langchain_openai": "langchain_openai",
    }
    
    results = {}
    for display_name, package_name in required_packages.items():
        try:
            __import__(package_name)
            results[display_name] = True
            print(f"  ✅ {display_name}")
        except ImportError:
            results[display_name] = False
            print(f"  ❌ {display_name}")
    
    return all(results.values())


def main():
    """Run all verifications"""
    print("=" * 70)
    print("🔍 PageIndex RAG Integration Verification")
    print("=" * 70)
    
    checks = [
        ("Dependencies", verify_dependencies),
        ("Imports", verify_imports),
        ("All RAG Types", verify_all_rag_types),
        ("CLI Dispatch", verify_cli_dispatch),
        ("Evaluator Instantiation", verify_evaluator_instantiation),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"\n⚠️  {check_name} check failed with exception: {e}")
            results[check_name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 70)
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:8} {check_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 All verification checks PASSED!")
        print("\nPageIndex RAG is successfully integrated.")
        print("\nYou can now run:")
        print("  python -m src.evaluation.ragas_evaluator pageindex")
        print("  python -m src.evaluation.ragas_evaluator all")
        print("  python -m src.evaluation.ragas_evaluator all-models-all-rags")
    else:
        print("⚠️  Some checks failed. See details above.")
        sys.exit(1)
    
    print("=" * 70)


if __name__ == "__main__":
    main()
