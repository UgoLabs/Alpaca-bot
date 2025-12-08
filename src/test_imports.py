try:
    print("Importing utils...")
    import utils
    print("Utils OK")
    
    print("Importing swing_env...")
    import swing_environment
    print("Swing Env OK")
    
    print("Importing paper_trade...")
    import paper_trade
    print("Paper Trade OK")

except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"General Error: {e}")
