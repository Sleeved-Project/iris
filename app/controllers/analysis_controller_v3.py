import os
import requests
from typing import List, Optional
import cv2
from PIL import Image
from sqlalchemy.orm import Session
from fastapi import Depends

# Suppression de l'import inutile :
# from app.services.contour_detection_service import detect_cards
from app.services.image_hash_service import image_hash_service
from app.dependencies.image_request_validators import ValidationResult
from app.schemas.analysis_schemas import (
    AnalysisResponse,
    AnalyzedCard,
    MatchedCardDetail,
)
from app.db.session import get_db
from app.db.models.card_hash import CardHash

# Remplacer ici par le nom correct de ton nouveau service d'extraction
from app.services.card_extraction_service_v3 import card_extraction_service

HASH_SIZE_DIMENSION = 16
BITS_PER_HASH = HASH_SIZE_DIMENSION * HASH_SIZE_DIMENSION
TOTAL_HASH_BITS = BITS_PER_HASH * 2

SIMILARITY_HAMMING_THRESHOLD = 90
TOP_N_RESULTS = 5


async def analyze_image(
    validated_input: ValidationResult,
    db: Session = Depends(get_db),
    debug: bool = False,
) -> AnalysisResponse:
    temp_image_path = None
    final_output_dir = "final_output"
    final_output_dir_low = "final_output2"
    os.makedirs(final_output_dir, exist_ok=True)
    os.makedirs(final_output_dir_low, exist_ok=True)

    try:
        if validated_input.file:
            contents = await validated_input.file.read()
            temp_image_path = f"/tmp/{validated_input.file.filename}"
            with open(temp_image_path, "wb") as f:
                f.write(contents)
        elif validated_input.url:
            raise NotImplementedError("Analysis from URL is not yet implemented.")

        if not temp_image_path or not os.path.exists(temp_image_path):
            raise ValueError("Image file not found for analysis.")

        extracted_cards = card_extraction_service.extract_cards_from_image(
            temp_image_path
        )

        if not extracted_cards or not isinstance(extracted_cards, list):
            raise ValueError("Extraction des cartes échouée : aucune carte extraite.")

        db_hashes = db.query(CardHash).all()
        analyzed_cards: List[AnalyzedCard] = []
        source_filename = os.path.basename(temp_image_path)

        for i in range(0, len(extracted_cards), 2):
            card_pair = extracted_cards[i : i + 2]

            # Initialisation des cartes la plus et moins similaire
            best_card_result = None
            best_similarity_percentage = -1.0  # plus bas que possible
            best_card_i = i

            worst_card_result = None
            worst_similarity_percentage = 101.0  # plus haut que possible
            worst_card_i = i

            for j, card_image_np in enumerate(card_pair):
                card_image_pil = Image.fromarray(
                    cv2.cvtColor(card_image_np, cv2.COLOR_BGR2RGB)
                )
                card_image_pil_grayscale = card_image_pil.convert("L")
                card_hash = image_hash_service.calculate_hashes(
                    card_image_pil_grayscale
                )

                best_match_id: Optional[str] = None
                best_match_name: Optional[str] = None
                min_distance = float("inf")
                is_similar = False
                similarity_percentage = 0.0

                all_matches: List[MatchedCardDetail] = []

                for db_card_hash_entry in db_hashes:
                    db_hash_str = db_card_hash_entry.hash
                    current_distance, current_percentage = (
                        image_hash_service.calculate_similarity_metrics(
                            card_hash, db_hash_str
                        )
                    )

                    if current_distance < min_distance:
                        min_distance = current_distance
                        best_match_id = db_card_hash_entry.id
                        best_match_name = getattr(
                            db_card_hash_entry, "name", f"Card {db_card_hash_entry.id}"
                        )
                        similarity_percentage = current_percentage

                    all_matches.append(
                        MatchedCardDetail(
                            card_id=db_card_hash_entry.id,
                            card_name=getattr(
                                db_card_hash_entry,
                                "name",
                                f"Card {db_card_hash_entry.id}",
                            ),
                            similarity_percentage=round(current_percentage, 2),
                            hamming_distance=current_distance,
                        )
                    )

                all_matches.sort(
                    key=lambda x: (x.similarity_percentage, -x.hamming_distance),
                    reverse=True,
                )

                top_n_results = all_matches[:TOP_N_RESULTS]
                if (
                    best_match_id is not None
                    and min_distance <= SIMILARITY_HAMMING_THRESHOLD
                ):
                    is_similar = True

                # Carte avec la plus forte similarité dans la paire
                if similarity_percentage > best_similarity_percentage:
                    best_similarity_percentage = similarity_percentage
                    best_card_result = AnalyzedCard(
                        card_hash=card_hash,
                        card_index=i // 2,
                        is_similar=is_similar,
                        similarity_percentage=round(similarity_percentage, 2),
                        matched_card_id=best_match_id,
                        matched_card_name=best_match_name,
                        top_n_matches=top_n_results,
                    )
                    best_card_i = i + j

                # Carte avec la plus faible similarité dans la paire
                if similarity_percentage < worst_similarity_percentage:
                    worst_similarity_percentage = similarity_percentage
                    worst_card_result = AnalyzedCard(
                        card_hash=card_hash,
                        card_index=i // 2,
                        is_similar=is_similar,
                        similarity_percentage=round(similarity_percentage, 2),
                        matched_card_id=best_match_id,
                        matched_card_name=best_match_name,
                        top_n_matches=top_n_results,
                    )
                    worst_card_i = i + j

            # Télécharger et sauvegarder la carte la plus similaire dans final_output
            if best_card_result and best_card_result.matched_card_id:
                try:
                    prefix, suffix = best_card_result.matched_card_id.split("-")
                    image_url = (
                        f"https://images.pokemontcg.io/{prefix}/{suffix}_hires.png"
                    )
                    if debug:
                        print(
                            f"""Lien vers la carte
                            {best_card_result.matched_card_id} : {image_url}"""
                        )

                    response = requests.get(image_url, stream=True)
                    if response.status_code == 200:
                        output_filename = (
                            f"card_{source_filename}_{best_card_i + 1}.png"
                        )
                        output_filepath = os.path.join(
                            final_output_dir, output_filename
                        )

                        with open(output_filepath, "wb") as f_out:
                            for chunk in response.iter_content(chunk_size=8192):
                                f_out.write(chunk)

                        if debug:
                            print(
                                f"""Image téléchargée et sauvegardée sous :
                                {output_filepath}"""
                            )
                    else:
                        print(
                            f"""Erreur téléchargement image {image_url}:
                            Status {response.status_code}"""
                        )
                except Exception as e:
                    print(
                        f"""Erreur lors du téléchargement ou sauvegarde
                        pour {best_card_result.matched_card_id}: {e}"""
                    )

            # Télécharger et sauvegarder la carte la moins similaire dans final_output2
            if worst_card_result and worst_card_result.matched_card_id:
                try:
                    prefix, suffix = worst_card_result.matched_card_id.split("-")
                    image_url = (
                        f"https://images.pokemontcg.io/{prefix}/{suffix}_hires.png"
                    )
                    if debug:
                        print(
                            f"""Lien vers la carte
                            {worst_card_result.matched_card_id} : {image_url}"""
                        )

                    response = requests.get(image_url, stream=True)
                    if response.status_code == 200:
                        output_filename = (
                            f"card_{source_filename}_{worst_card_i + 1}.png"
                        )
                        output_filepath = os.path.join(
                            final_output_dir_low, output_filename
                        )

                        with open(output_filepath, "wb") as f_out:
                            for chunk in response.iter_content(chunk_size=8192):
                                f_out.write(chunk)

                        if debug:
                            print(
                                f"""Image téléchargée et sauvegardée sous :
                                {output_filepath}"""
                            )
                    else:
                        print(
                            f"""Erreur téléchargement image {image_url}:
                            Status {response.status_code}"""
                        )
                except Exception as e:
                    print(
                        f"""Erreur lors du téléchargement ou sauvegarde pour
                        {worst_card_result.matched_card_id}: {e}"""
                    )

            # On peut choisir ici d'ajouter uniquement la carte
            # la plus similaire, la moins similaire, ou les deux.
            # Ici je les ajoute toutes les deux pour suivi.
            if best_card_result:
                analyzed_cards.append(best_card_result)
            """ if worst_card_result and worst_card_result != best_card_result:
                analyzed_cards.append(worst_card_result) """

        return AnalysisResponse(
            message="""Analyse complétée avec succès via le nouveau modèle
            d'extraction (cartes aux similarités les plus haute
            et basse par paire sauvegardées).""",
            cards=analyzed_cards,
        )

    except Exception as e:
        print(f"Erreur durant l'analyse : {e}")
        raise e
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
