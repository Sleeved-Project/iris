import asyncio
import json
import logging
import sys
from pathlib import Path

from app.db.models.card_hash import CardHash
from app.db.session import SessionLocal, init_db
from app.services.image_hash_service import image_hash_service

# Add project root to path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


async def import_cards(json_file=None, small_images=True):
    """
    Import card hashes from JSON file into database.

    Args:
        json_file: Path to the JSON file containing card data
        small_images: Whether to use small images (faster) or large images
            (better quality)
    """
    # Set default JSON file path if not provided
    if json_file is None:
        json_file = Path(__file__).parent / "cards_data.json"

    logging.info(f"Reading cards from {json_file}")

    # Initialize the database if needed
    init_db()

    # Open JSON file
    try:
        with open(json_file, "r") as f:
            cards = json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {json_file}")
        return

    # Create session
    db = SessionLocal()

    try:
        # Get existing cards to avoid duplicates
        existing_cards = {card.id for card in db.query(CardHash.id).all()}

        for i, card in enumerate(cards):
            card_id = card["id"]

            # Skip if already exists
            if card_id in existing_cards:
                logging.info(f"Card {card_id} already exists in database, skipping...")
                continue

            # Choose image URL based on size preference
            image_url = card["image_small"] if small_images else card["image_large"]

            try:
                logging.info(f"Processing card {i+1}/{len(cards)}: {card_id}")

                # Calculate hash using our service
                hash_value = await image_hash_service.hash_from_url(image_url)

                # Store in database
                card_hash = CardHash(id=card_id, hash=hash_value)
                db.add(card_hash)
                db.commit()

                logging.info(f"Saved hash for {card_id}")

            except Exception as e:
                db.rollback()
                logging.error(f"Error processing card {card_id}: {str(e)}")
                continue

    except Exception as e:
        logging.error(f"Error during import: {str(e)}")
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Get command line argument for image size
    use_small = True
    if len(sys.argv) > 1 and sys.argv[1].lower() == "large":
        use_small = False
        logging.info("Using large images (slower but higher quality)")
    else:
        logging.info("Using small images (faster)")

    asyncio.run(import_cards(small_images=use_small))
    logging.info("Import completed")
