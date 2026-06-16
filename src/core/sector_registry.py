"""Registry of ~136 sector profiles covering the French/EU job market.

SECTORS       — sector_key → [profile_ids]  (as specified)
ALL_PROFILES  — profile_id → SectorProfile  (detection + criteria)
GENERIC_PROFILE — fallback when confidence < 0.3
SECTOR_DISPLAY_NAMES — sector_key → human-readable label (for UI)
PROFILE_BY_SECTOR    — display_label → [profile_ids]  (for Phase C dropdown)
"""

from src.core.sector_profiles import Criterion, SectorProfile

# ---------------------------------------------------------------------------
# Human-readable sector labels
# ---------------------------------------------------------------------------

SECTOR_DISPLAY_NAMES: dict[str, str] = {
    "industrie_manufacturiere": "Industrie & Fabrication",
    "btp": "BTP & Construction",
    "agroalimentaire": "Agroalimentaire",
    "energie_environnement": "Énergie & Environnement",
    "commerce_distribution": "Commerce & Distribution",
    "transport_logistique": "Transport & Logistique",
    "hotellerie_restauration": "Hôtellerie & Restauration",
    "sante_social": "Santé & Social",
    "education_formation": "Éducation & Formation",
    "securite": "Sécurité",
    "nettoyage_services": "Nettoyage & Services",
    "coiffure_esthetique": "Coiffure & Esthétique",
    "immobilier": "Immobilier",
    "banque_assurance": "Banque & Assurance",
    "finance_comptabilite": "Finance & Comptabilité",
    "rh_recrutement": "RH & Recrutement",
    "juridique_administratif": "Juridique & Administratif",
    "marketing_communication": "Marketing & Communication",
    "informatique_digital": "Informatique & Digital",
    "tourisme_loisirs": "Tourisme & Loisirs",
    "sport": "Sport",
    "culture_medias": "Culture & Médias",
    "artisanat": "Artisanat",
}

# ---------------------------------------------------------------------------
# Profile data — (id, sector_key, job_title, aliases, detection_keywords)
# Aliases: ≥3 FR variants + 1-2 EN.
# Keywords: specific, discriminant — never generic single words.
# ---------------------------------------------------------------------------

_PROFILE_SPECS: list[tuple] = [

    # =========================================================
    # INDUSTRIE & FABRICATION
    # =========================================================

    (
        "operateur_production", "industrie_manufacturiere",
        "Opérateur de production",
        ["opérateur de production", "opérateur sur machine", "agent de production",
         "opérateur fabrication", "production operator", "manufacturing operator"],
        ["cadence de production", "gamme de fabrication", "chaîne de montage",
         "poste de travail", "rendement", "plan de production"],
    ),
    (
        "conducteur_ligne", "industrie_manufacturiere",
        "Conducteur de ligne",
        ["conducteur de ligne", "conducteur d'installation", "conducteur de machine",
         "chef de ligne", "line operator", "production line operator"],
        ["conduite de ligne", "réglage machine", "automate industriel",
         "maintenance 1er niveau", "arrêt de ligne"],
    ),
    (
        "regleur_machine", "industrie_manufacturiere",
        "Régleur sur machine",
        ["régleur", "régleur sur machine", "régleur-monteur", "technicien de réglage",
         "machine setter", "cnc setter"],
        ["réglage outillage", "programme CNC", "paramètres machine",
         "gamme de réglage", "temps de cycle"],
    ),
    (
        "technicien_maintenance", "industrie_manufacturiere",
        "Technicien de maintenance industrielle",
        ["technicien de maintenance", "technicien maintenance industrielle",
         "technicien de maintenance industrielle", "maintenance technician",
         "maintenance engineer"],
        ["GMAO", "maintenance préventive", "maintenance corrective",
         "dépannage électromécanique", "pneumatique", "hydraulique"],
    ),
    (
        "controleur_qualite", "industrie_manufacturiere",
        "Contrôleur qualité",
        ["contrôleur qualité", "technicien qualité", "inspecteur qualité",
         "agent qualité", "quality controller", "quality inspector"],
        ["contrôle dimensionnel", "métrologie", "ISO 9001",
         "non-conformité", "plan de contrôle", "pied à coulisse"],
    ),
    (
        "soudeur", "industrie_manufacturiere",
        "Soudeur",
        ["soudeur", "soudeur assembleur", "soudeur-monteur",
         "soudeur industriel", "welder", "welding technician"],
        ["soudure MIG", "soudure TIG", "soudure MAG",
         "cordon de soudure", "mode opératoire soudage", "baguette de soudure"],
    ),
    (
        "usineur", "industrie_manufacturiere",
        "Usineur",
        ["usineur", "tourneur", "fraiseur", "tourneur-fraiseur",
         "machinist", "cnc machinist"],
        ["usinage CNC", "tour à commande numérique", "fraiseuse",
         "tolérance dimensionnelle", "plan industriel", "cote"],
    ),
    (
        "electromecanicien", "industrie_manufacturiere",
        "Électromécanicien",
        ["électromécanicien", "électrotechnicien", "technicien électromécanique",
         "électromécanicien industriel", "electromechanical technician",
         "electromechanical engineer"],
        ["schéma électrique", "armoire électrique", "variateur de fréquence",
         "moteur triphasé", "automate programmable", "lecture de schémas"],
    ),
    (
        "agent_fabrication", "industrie_manufacturiere",
        "Agent de fabrication",
        ["agent de fabrication", "agent de production", "opérateur de montage",
         "agent montage assemblage", "assembly operator", "production worker"],
        ["assemblage manuel", "contrôle visuel", "fiche de suivi production",
         "bon de fabrication", "cadence horaire"],
    ),

    # =========================================================
    # BTP & CONSTRUCTION
    # =========================================================

    (
        "macon", "btp",
        "Maçon",
        ["maçon", "maçon traditionnel", "maçon bâtiment",
         "maçon coffreur", "mason", "bricklayer"],
        ["parpaing", "coffrage béton", "coulage béton",
         "enduit façade", "élévation de murs", "fondation"],
    ),
    (
        "electricien_batiment", "btp",
        "Électricien du bâtiment",
        ["électricien bâtiment", "électricien du bâtiment", "électricien",
         "électricien d'installation", "electrician", "electrical installer"],
        ["câblage électrique", "tableau électrique", "habilitation électrique",
         "gaine technique", "installation électrique", "norme NF C 15-100"],
    ),
    (
        "plombier", "btp",
        "Plombier-chauffagiste",
        ["plombier", "plombier-chauffagiste", "technicien plomberie",
         "installateur sanitaire", "plumber", "heating engineer"],
        ["tuyauterie cuivre", "robinetterie sanitaire", "chaudière gaz",
         "soudure plomberie", "installation sanitaire", "norme gaz"],
    ),
    (
        "carreleur", "btp",
        "Carreleur",
        ["carreleur", "poseur de carrelage", "carreleur-faïencier",
         "poseur revêtement", "tiler", "tile fitter"],
        ["pose carrelage", "ragréage", "mortier-colle",
         "joints carrelage", "faïence murale", "niveaux et aplombs"],
    ),
    (
        "peintre_batiment", "btp",
        "Peintre en bâtiment",
        ["peintre bâtiment", "peintre en bâtiment", "peintre-décorateur",
         "peintre revêtements", "painter", "decorator"],
        ["préparation support peinture", "enduit de lissage", "lasure bois",
         "ravalement façade", "peinture intérieure", "impression"],
    ),
    (
        "couvreur", "btp",
        "Couvreur",
        ["couvreur", "couvreur-zingueur", "technicien couverture",
         "couvreur charpentier", "roofer", "roofing contractor"],
        ["pose tuiles", "zinguerie", "étanchéité toiture",
         "charpente bois", "gouttière", "ardoise"],
    ),
    (
        "menuisier", "btp",
        "Menuisier",
        ["menuisier", "menuisier-ébéniste", "menuisier poseur",
         "menuisier bois", "carpenter", "joiner"],
        ["menuiserie bois", "assemblage tenon mortaise", "pose fenêtres",
         "portes intérieures", "machines à bois", "atelier bois"],
    ),
    (
        "charpentier", "btp",
        "Charpentier",
        ["charpentier", "charpentier bois", "charpentier-couvreur",
         "charpentier métallique", "timber framer", "structural carpenter"],
        ["charpente traditionnelle", "ossature bois", "levage charpente",
         "taille bois", "structures bois", "poutres"],
    ),
    (
        "chef_chantier", "btp",
        "Chef de chantier",
        ["chef de chantier", "chef chantier", "responsable chantier",
         "conducteur de travaux terrain", "site supervisor", "construction foreman"],
        ["planification chantier", "PPSPS", "réunion de chantier",
         "coordination sous-traitants", "gestion main d'œuvre", "suivi travaux"],
    ),
    (
        "conducteur_travaux", "btp",
        "Conducteur de travaux",
        ["conducteur de travaux", "conducteur travaux", "responsable travaux",
         "chargé de travaux", "construction project manager", "site manager"],
        ["CCTP", "DPGF", "planning travaux", "réception de chantier",
         "devis et métrés", "budget travaux"],
    ),

    # =========================================================
    # AGROALIMENTAIRE
    # =========================================================

    (
        "operateur_agroalimentaire", "agroalimentaire",
        "Opérateur en agroalimentaire",
        ["opérateur agroalimentaire", "agent de production agroalimentaire",
         "opérateur industrie alimentaire", "opérateur agro",
         "food production operator", "food manufacturing worker"],
        ["BPH", "HACCP agroalimentaire", "traçabilité alimentaire",
         "chaîne du froid", "hygiène alimentaire", "nettoyage désinfection"],
    ),
    (
        "agent_conditionnement", "agroalimentaire",
        "Agent de conditionnement",
        ["agent de conditionnement", "opérateur de conditionnement",
         "conditionneur", "agent emballage", "packaging operator", "packing operative"],
        ["ligne de conditionnement", "étiquetage produits", "palettisation",
         "contrôle poids", "operculeuse", "emballage alimentaire"],
    ),
    (
        "technicien_qualite_alimentaire", "agroalimentaire",
        "Technicien qualité agroalimentaire",
        ["technicien qualité alimentaire", "technicien QHSE alimentaire",
         "responsable qualité agroalimentaire", "food quality technician",
         "food safety technician"],
        ["HACCP", "IFS Food", "BRC Global", "analyse microbiologique",
         "plan de maîtrise sanitaire", "audit fournisseurs alimentaires"],
    ),
    (
        "chef_production_agroalimentaire", "agroalimentaire",
        "Chef de production agroalimentaire",
        ["chef de production agroalimentaire", "responsable de production agro",
         "directeur de production alimentaire", "production manager food",
         "food production manager"],
        ["rendement matière première", "planification production alimentaire",
         "certification IFS", "gestion équipe production", "coûts de production"],
    ),

    # =========================================================
    # ÉNERGIE & ENVIRONNEMENT
    # =========================================================

    (
        "technicien_eolien", "energie_environnement",
        "Technicien de maintenance éolienne",
        ["technicien éolien", "technicien maintenance éolienne",
         "monteur éolien", "technicien turbine", "wind turbine technician",
         "wind energy technician"],
        ["nacelle éolienne", "certification GWO", "travaux en hauteur éolien",
         "maintenance rotor", "gréage éolien", "turbine de vent"],
    ),
    (
        "electricien_industriel", "energie_environnement",
        "Électricien industriel",
        ["électricien industriel", "électricien d'industrie",
         "monteur câbleur industriel", "industrial electrician",
         "electrical engineer"],
        ["habilitation électrique B2", "armoire électrique industrielle",
         "variateur de fréquence", "automate Siemens", "moteur triphasé"],
    ),
    (
        "agent_traitement_eaux", "energie_environnement",
        "Agent de traitement des eaux",
        ["agent de traitement des eaux", "opérateur station d'épuration",
         "technicien STEP", "water treatment operator", "water treatment technician"],
        ["station d'épuration", "potabilisation eau", "analyses eau",
         "boues biologiques", "bassin d'aération", "réglementation eau"],
    ),
    (
        "technicien_environnement", "energie_environnement",
        "Technicien environnement",
        ["technicien environnement", "chargé d'environnement",
         "technicien HSE", "environmental technician", "hse officer"],
        ["ISO 14001", "bilan carbone", "audit environnemental",
         "gestion déchets", "réglementation ICPE", "impact environnemental"],
    ),

    # =========================================================
    # COMMERCE & DISTRIBUTION
    # =========================================================

    (
        "vendeur", "commerce_distribution",
        "Vendeur conseil",
        ["vendeur", "vendeur conseil", "conseillère de vente",
         "vendeur en magasin", "sales advisor", "sales associate",
         "retail sales assistant"],
        ["encaissement", "facing", "merchandising",
         "point de vente", "fidélisation clientèle", "rayon"],
    ),
    (
        "caissier", "commerce_distribution",
        "Caissier",
        ["caissier", "caissière", "hôte de caisse",
         "hôtesse de caisse", "cashier", "checkout operator"],
        ["tenue de caisse", "rendu monnaie", "gestion de caisse",
         "caisse enregistreuse", "carte fidélité", "flux de caisse"],
    ),
    (
        "conseiller_vente", "commerce_distribution",
        "Conseiller de vente",
        ["conseiller de vente", "conseiller commercial", "conseiller clientèle",
         "commercial sédentaire", "sales consultant", "sales representative"],
        ["techniques de vente", "argumentation commerciale", "négociation client",
         "devis commercial", "suivi portefeuille clients", "satisfaction client"],
    ),
    (
        "responsable_rayon", "commerce_distribution",
        "Responsable de rayon",
        ["responsable de rayon", "chef de rayon", "manager rayon",
         "responsable département", "department manager", "section manager"],
        ["gestion de rayon", "réapprovisionnement", "inventaire tournant",
         "marge brute", "CA rayon", "animation commerciale rayon"],
    ),
    (
        "directeur_magasin", "commerce_distribution",
        "Directeur de magasin",
        ["directeur de magasin", "responsable de magasin", "gérant de magasin",
         "directeur de site commercial", "store manager", "retail manager"],
        ["compte d'exploitation magasin", "animation équipes vente",
         "résultats commerciaux", "budget magasin", "recrutement équipe"],
    ),
    (
        "merchandiser", "commerce_distribution",
        "Merchandiser",
        ["merchandiser", "animateur commercial", "visual merchandiser",
         "animateur merchandising", "trade marketing", "retail merchandiser"],
        ["planogramme", "implantation rayon", "vitrine commerciale",
         "opération commerciale", "mise en avant produit", "préconisations merchandising"],
    ),
    (
        "responsable_ecommerce", "commerce_distribution",
        "Responsable e-commerce",
        ["responsable e-commerce", "e-commerce manager", "gestionnaire boutique en ligne",
         "digital commerce manager", "online store manager"],
        ["taux de conversion", "panier moyen", "Google Analytics e-commerce",
         "marketplace", "CRO", "SEO boutique en ligne"],
    ),
    (
        "acheteur", "commerce_distribution",
        "Acheteur",
        ["acheteur", "acheteur-approvisionneur", "acheteur junior",
         "responsable achats", "buyer", "purchasing manager"],
        ["appels d'offres fournisseurs", "négociation achats", "panel fournisseurs",
         "référencement produits", "conditions d'achat", "cahier des charges achat"],
    ),

    # =========================================================
    # TRANSPORT & LOGISTIQUE
    # =========================================================

    (
        "chauffeur_pl", "transport_logistique",
        "Chauffeur poids lourds",
        ["chauffeur PL", "chauffeur poids lourds", "chauffeur camion",
         "conducteur routier", "HGV driver", "truck driver"],
        ["permis C", "FIMO marchandises", "FCO transport",
         "chronotachygraphe", "bons de livraison", "tournée PL"],
    ),
    (
        "chauffeur_spl", "transport_logistique",
        "Chauffeur SPL",
        ["chauffeur SPL", "chauffeur super poids lourds", "chauffeur semi-remorque",
         "conducteur SPL", "articulated lorry driver", "semi-truck driver"],
        ["permis CE", "FIMO SPL", "semi-remorque",
         "ADR marchandises dangereuses", "chronotachygraphe", "FCO SPL"],
    ),
    (
        "conducteur_livreur", "transport_logistique",
        "Conducteur-livreur",
        ["conducteur-livreur", "livreur", "chauffeur-livreur",
         "agent de livraison", "delivery driver", "courier driver"],
        ["tournée de livraison", "bon de livraison", "manutention colis",
         "permis B", "optimisation tournée", "scan livraison"],
    ),
    (
        "gestionnaire_stock", "transport_logistique",
        "Gestionnaire de stocks",
        ["gestionnaire de stock", "responsable des stocks", "gestionnaire magasin",
         "magasinier responsable", "stock manager", "inventory manager"],
        ["WMS", "inventaire tournant", "méthode FIFO", "méthode FEFO",
         "taux de service", "référence article stock"],
    ),
    (
        "preparateur_commandes", "transport_logistique",
        "Préparateur de commandes",
        ["préparateur de commandes", "agent de préparation", "picking",
         "préparateur logistique", "order picker", "warehouse picker"],
        ["CACES 1A", "préparation de commandes vocale", "scan code-barres",
         "bon de commande entrepôt", "palettisation commandes", "zone de picking"],
    ),
    (
        "cariste", "transport_logistique",
        "Cariste",
        ["cariste", "conducteur de chariot élévateur", "magasinier cariste",
         "opérateur chariot", "forklift operator", "forklift driver"],
        ["CACES R489", "chariot élévateur frontal", "gerbage en hauteur",
         "zone de stockage entrepôt", "chariots R489 catégorie 3"],
    ),
    (
        "responsable_logistique", "transport_logistique",
        "Responsable logistique",
        ["responsable logistique", "directeur logistique", "supply chain manager",
         "responsable supply chain", "logistics manager", "supply chain director"],
        ["supply chain", "KPI logistique", "optimisation flux entrepôt",
         "coût logistique", "WMS déploiement", "prestataires logistiques"],
    ),
    (
        "affreteur", "transport_logistique",
        "Affréteur",
        ["affréteur", "affréteur routier", "courtier en transport",
         "chargé d'affrètement", "freight forwarder", "freight broker"],
        ["bourse de fret", "cotation transport routier", "négociation transporteurs",
         "incoterms", "affrètement", "cahier des charges transport"],
    ),
    (
        "agent_exploitation_transport", "transport_logistique",
        "Agent d'exploitation transport",
        ["agent d'exploitation transport", "exploitant transport", "planificateur transport",
         "dispatcher", "transport planner", "transport coordinator"],
        ["planification tournées", "TMS", "réglementation transport routier",
         "gestion chauffeurs", "affrètement ponctuel", "temps de service"],
    ),
    (
        "declarant_douane", "transport_logistique",
        "Déclarant en douane",
        ["déclarant en douane", "agent douanier", "déclarant douanier",
         "responsable douane", "customs agent", "customs broker"],
        ["dédouanement", "régime douanier", "DAU douane",
         "tarif douanier commun", "DEB DES", "incoterms douane"],
    ),

    # =========================================================
    # HÔTELLERIE & RESTAURATION
    # =========================================================

    (
        "serveur", "hotellerie_restauration",
        "Serveur en restaurant",
        ["serveur", "serveur en restaurant", "serveur en salle",
         "commis de salle", "waiter", "waitress", "food server"],
        ["service en salle", "mise en place restaurant", "prise de commande",
         "découpe devant le client", "carte des vins", "service à l'assiette"],
    ),
    (
        "cuisinier", "hotellerie_restauration",
        "Cuisinier",
        ["cuisinier", "commis de cuisine", "cuisinier de collectivité",
         "aide-cuisinier", "cook", "kitchen worker"],
        ["préparations culinaires", "fiche technique recette", "taillage légumes",
         "brigade de cuisine", "HACCP cuisine", "poste chaud"],
    ),
    (
        "chef_partie", "hotellerie_restauration",
        "Chef de partie",
        ["chef de partie", "demi-chef de partie", "chef saucier",
         "chef garde-manger", "chef de partie cuisine", "station chef", "line cook"],
        ["gestion de poste cuisine", "encadrement commis", "fiche de poste brigade",
         "entrées chaudes", "poissons viandes", "production culinaire"],
    ),
    (
        "chef_cuisinier", "hotellerie_restauration",
        "Chef cuisinier",
        ["chef cuisinier", "chef de cuisine", "chef exécutif",
         "directeur de cuisine", "executive chef", "head chef"],
        ["création carte restaurant", "food cost", "gestion brigade complète",
         "commandes fournisseurs restauration", "ratio matière", "menu engineering"],
    ),
    (
        "receptionniste", "hotellerie_restauration",
        "Réceptionniste d'hôtel",
        ["réceptionniste d'hôtel", "agent de réception", "réceptionniste",
         "hôtesse d'accueil hôtel", "hotel receptionist", "front desk agent"],
        ["check-in check-out", "logiciel PMS hôtel", "réservations Opera",
         "accueil clientèle hôtelière", "Fidelio hôtel", "planning chambres"],
    ),
    (
        "barman", "hotellerie_restauration",
        "Barman",
        ["barman", "barmaid", "bartender",
         "responsable bar", "mixologiste", "bar staff"],
        ["mixologie", "cocktails classiques", "carte des boissons",
         "gestion stocks bar", "service comptoir", "technique de shaker"],
    ),
    (
        "maitre_hotel", "hotellerie_restauration",
        "Maître d'hôtel",
        ["maître d'hôtel", "responsable de salle", "directeur de salle",
         "chef de rang", "head waiter", "restaurant floor manager"],
        ["service de salle gastronomique", "découpe flambage", "sommellerie",
         "formation équipe salle", "réservations restaurant", "gestion salle"],
    ),
    (
        "gouvernante", "hotellerie_restauration",
        "Gouvernante d'hôtel",
        ["gouvernante d'hôtel", "gouvernante d'étage", "responsable étages",
         "superviseur ménage", "housekeeper", "housekeeping supervisor"],
        ["contrôle ménage chambres", "lingerie hôtelière", "rapport gouvernante",
         "équipe de chambre", "normes propreté hôtel", "inventaire linge"],
    ),
    (
        "directeur_restaurant", "hotellerie_restauration",
        "Directeur de restaurant",
        ["directeur de restaurant", "gérant de restaurant", "responsable restaurant",
         "directeur de point de vente restauration", "restaurant manager",
         "food and beverage manager"],
        ["compte d'exploitation restaurant", "food cost ratio",
         "gestion équipe restauration", "hygiène HACCP restaurant",
         "fournisseurs restauration", "CA restaurant"],
    ),

    # =========================================================
    # SANTÉ & SOCIAL
    # =========================================================

    (
        "aide_soignant", "sante_social",
        "Aide-soignant",
        ["aide-soignant", "aide soignante", "assistant de soins",
         "auxiliaire de soins", "nursing assistant", "healthcare assistant",
         "care assistant"],
        ["soins d'hygiène corporelle", "toilette au lit", "aide à la mobilisation",
         "transmissions aide-soignant", "EHPAD", "diurèse"],
    ),
    (
        "infirmier", "sante_social",
        "Infirmier diplômé d'État",
        ["infirmier", "infirmière diplômée d'état", "IDE", "infirmier DE",
         "registered nurse", "nurse", "rn"],
        ["soins infirmiers", "perfusion intraveineuse", "pansements complexes",
         "prescriptions médicales", "transmissions infirmières", "protocole de soins"],
    ),
    (
        "medecin", "sante_social",
        "Médecin",
        ["médecin", "médecin généraliste", "docteur en médecine",
         "médecin spécialiste", "physician", "general practitioner", "doctor"],
        ["diagnostic médical", "ordonnance médicale", "consultation médicale",
         "RPPS", "garde hospitalière", "dossier patient médical"],
    ),
    (
        "pharmacien", "sante_social",
        "Pharmacien",
        ["pharmacien", "pharmacien d'officine", "pharmacien hospitalier",
         "pharmacien biologiste", "pharmacist", "dispensing pharmacist"],
        ["délivrance médicaments", "ordonnancier", "pharmacovigilance",
         "conseil pharmaceutique", "RPPS pharmacien", "stupéfiants"],
    ),
    (
        "kinesitherapeute", "sante_social",
        "Kinésithérapeute",
        ["kinésithérapeute", "masseur-kinésithérapeute", "kiné",
         "rééducateur kinésithérapeute", "physiotherapist", "physical therapist"],
        ["rééducation fonctionnelle", "bilan kinésithérapique", "massage thérapeutique",
         "RPPS kinésithérapie", "rééducation post-opératoire", "électrothérapie"],
    ),
    (
        "aide_domicile", "sante_social",
        "Aide à domicile",
        ["aide à domicile", "auxiliaire de vie sociale", "AVS",
         "assistant de vie", "home care assistant", "personal care assistant"],
        ["maintien à domicile", "aide à la toilette domicile", "aide aux repas",
         "bénéficiaire dépendant", "SAAD", "plan d'aide"],
    ),
    (
        "educateur_specialise", "sante_social",
        "Éducateur spécialisé",
        ["éducateur spécialisé", "éducateur de jeunes enfants", "éducateur ES",
         "moniteur éducateur", "special educator", "support worker"],
        ["DEEES", "accompagnement éducatif", "projet personnalisé d'accompagnement",
         "ESMS", "mesure éducative judiciaire", "travail en équipe pluridisciplinaire"],
    ),
    (
        "assistant_social", "sante_social",
        "Assistant de service social",
        ["assistant social", "assistante sociale", "travailleur social",
         "conseiller en économie sociale", "social worker", "welfare officer"],
        ["DEASS", "droit social", "accompagnement social global",
         "RSA CAF", "mesures sociales", "rapport social"],
    ),
    (
        "psychologue", "sante_social",
        "Psychologue",
        ["psychologue", "psychologue clinicien", "psychologue du travail",
         "psychologue scolaire", "psychologist", "clinical psychologist"],
        ["bilan psychologique", "entretien clinique", "tests psychométriques",
         "WISC WAIS", "thérapie cognitivo-comportementale", "RPPS psychologue"],
    ),
    (
        "orthophoniste", "sante_social",
        "Orthophoniste",
        ["orthophoniste", "logopède", "rééducateur du langage",
         "orthophoniste libéral", "speech therapist", "speech and language therapist"],
        ["bilan orthophonique", "rééducation dyslexie", "aphasie",
         "déglutition", "RPPS orthophoniste", "retard de parole"],
    ),

    # =========================================================
    # ÉDUCATION & FORMATION
    # =========================================================

    (
        "enseignant", "education_formation",
        "Enseignant",
        ["enseignant", "professeur", "professeur des écoles",
         "enseignant certifié", "teacher", "educator", "instructor"],
        ["CAPES", "agrégation", "progression pédagogique annuelle",
         "conseil de classe", "Éducation Nationale", "concours enseignement"],
    ),
    (
        "formateur", "education_formation",
        "Formateur professionnel",
        ["formateur", "formateur professionnel", "consultant formateur",
         "formateur indépendant", "trainer", "learning facilitator"],
        ["animation de formation", "ingénierie pédagogique", "Qualiopi",
         "OPCO", "e-learning", "séquence pédagogique"],
    ),
    (
        "animateur_bafa", "education_formation",
        "Animateur périscolaire",
        ["animateur BAFA", "animateur périscolaire", "animateur jeunesse",
         "animateur centre de loisirs", "youth worker", "activity leader"],
        ["BAFA", "ACM accueil collectif mineurs", "programme animation enfants",
         "TAP temps activités périscolaires", "centre de loisirs", "ATSEM"],
    ),
    (
        "moniteur_auto_ecole", "education_formation",
        "Moniteur d'auto-école",
        ["moniteur auto-école", "enseignant de la conduite", "moniteur conduite",
         "formateur conduite", "driving instructor", "road safety instructor"],
        ["BEPECASER", "code de la route", "conduite accompagnée AAC",
         "examen permis B", "livret d'apprentissage conducteur"],
    ),
    (
        "conseiller_orientation", "education_formation",
        "Conseiller en insertion professionnelle",
        ["conseiller d'orientation", "conseiller en insertion professionnelle",
         "conseiller emploi formation", "careers advisor", "employment counselor"],
        ["bilan de compétences", "insertion professionnelle",
         "France Travail accompagnement", "VAE validation des acquis", "CIO"],
    ),

    # =========================================================
    # SÉCURITÉ
    # =========================================================

    (
        "agent_securite", "securite",
        "Agent de sécurité",
        ["agent de sécurité", "agent de prévention et sécurité", "vigile",
         "agent de surveillance", "security officer", "security guard"],
        ["CQP APS", "carte professionnelle CNAPS", "ronde de sécurité",
         "contrôle d'accès", "main courante sécurité", "levée de doute"],
    ),
    (
        "agent_surete_aeroportuaire", "securite",
        "Agent de sûreté aéroportuaire",
        ["agent de sûreté aéroportuaire", "agent sûreté aéroport",
         "inspecteur sûreté", "airport security agent", "aviation security officer"],
        ["criblage passagers", "palpation sécuritaire", "zone réservée aéroportuaire",
         "habilitation aéroportuaire", "CNAPS aéroport", "TEH"],
    ),
    (
        "responsable_securite", "securite",
        "Responsable sécurité",
        ["responsable sécurité", "directeur sécurité", "chef de service sécurité",
         "responsable sûreté", "security manager", "head of security"],
        ["plan de sécurité", "gestion de crise", "audit de sécurité",
         "formation sécurité incendie", "ERP établissement recevant du public", "DUERP"],
    ),

    # =========================================================
    # NETTOYAGE & SERVICES
    # =========================================================

    (
        "agent_entretien", "nettoyage_services",
        "Agent d'entretien",
        ["agent d'entretien", "agent de nettoyage", "agent de propreté",
         "technicien de surface", "cleaning agent", "cleaner", "janitor"],
        ["plan de nettoyage désinfection", "produits biocides",
         "fiche de données sécurité", "monobrosse lustrage",
         "injection extraction moquette"],
    ),
    (
        "chef_equipe_nettoyage", "nettoyage_services",
        "Chef d'équipe propreté",
        ["chef d'équipe nettoyage", "responsable de site propreté",
         "encadrant propreté", "cleaning team leader", "cleaning supervisor"],
        ["gestion équipe propreté", "planning nettoyage",
         "contrôle qualité propreté", "formation agents nettoyage",
         "cahier des charges propreté"],
    ),

    # =========================================================
    # COIFFURE & ESTHÉTIQUE
    # =========================================================

    (
        "coiffeur", "coiffure_esthetique",
        "Coiffeur",
        ["coiffeur", "coiffeur mixte", "styliste coiffeur",
         "coiffeur-coloriste", "hairdresser", "hair stylist"],
        ["coupe femme", "coloration végétale", "balayage mèches",
         "permanente", "brushing", "CAP coiffure"],
    ),
    (
        "estheticienne", "coiffure_esthetique",
        "Esthéticienne",
        ["esthéticienne", "praticienne esthétique", "conseillère beauté",
         "esthéticienne cosméticienne", "beautician", "beauty therapist"],
        ["soins visage", "épilation cire", "manucure pose gel",
         "UV cabine", "brevet professionnel esthétique", "soins corporels"],
    ),
    (
        "prothesiste_ongulaire", "coiffure_esthetique",
        "Prothésiste ongulaire",
        ["prothésiste ongulaire", "technicienne ongles", "nail artist",
         "poseur capsules ongles", "nail technician", "nail specialist"],
        ["pose capsules gel UV", "résine ongles", "nail art",
         "chablons ongulaires", "remplissage gel", "faux ongles"],
    ),

    # =========================================================
    # IMMOBILIER
    # =========================================================

    (
        "agent_immobilier", "immobilier",
        "Agent immobilier",
        ["agent immobilier", "agent immobilier mandataire", "négociateur immobilier",
         "conseiller immobilier", "real estate agent", "property agent"],
        ["carte T immobilier", "compromis de vente", "mandat exclusif",
         "estimation immobilière", "diagnostics DPE", "réseau agence immobilière"],
    ),
    (
        "gestionnaire_locatif", "immobilier",
        "Gestionnaire locatif",
        ["gestionnaire locatif", "property manager", "responsable gestion locative",
         "chargé de gestion locative", "lettings manager", "rental manager"],
        ["quittancement loyers", "bail d'habitation", "état des lieux contradictoire",
         "gestion locative", "charges locatives", "dépôt de garantie"],
    ),
    (
        "syndic_copropriete", "immobilier",
        "Gestionnaire de copropriété",
        ["syndic de copropriété", "gestionnaire copropriété", "responsable syndicat",
         "chargé de copropriété", "property management", "building manager"],
        ["assemblée générale copropriété", "budget prévisionnel copropriété",
         "charges de copropriété", "règlement de copropriété", "carnet d'entretien immeuble"],
    ),
    (
        "negociateur_immobilier", "immobilier",
        "Négociateur immobilier",
        ["négociateur immobilier", "commercial immobilier", "chasseur immobilier",
         "conseiller transaction", "real estate negotiator", "property consultant"],
        ["prospection terrain immobilier", "rentrée de mandats", "pige immobilière",
         "visite de biens", "négociation prix immobilier", "compromis"],
    ),

    # =========================================================
    # BANQUE & ASSURANCE
    # =========================================================

    (
        "conseiller_bancaire", "banque_assurance",
        "Conseiller bancaire",
        ["conseiller bancaire", "conseiller de clientèle banque",
         "conseiller financier particulier", "bank advisor", "financial advisor"],
        ["portefeuille clients bancaire", "produits bancaires dépôts",
         "épargne réglementée", "crédit immobilier bancaire",
         "assurance vie bancassurance", "conformité KYC"],
    ),
    (
        "gestionnaire_patrimoine", "banque_assurance",
        "Conseiller en gestion de patrimoine",
        ["gestionnaire de patrimoine", "conseiller en gestion de patrimoine CGP",
         "conseiller patrimonial", "wealth manager", "financial planner"],
        ["CGP", "allocation d'actifs", "fiscalité patrimoniale",
         "SCPI", "IFI", "optimisation successorale"],
    ),
    (
        "actuaire", "banque_assurance",
        "Actuaire",
        ["actuaire", "actuaire assurance", "analyste actuariel",
         "actuaire pricing", "actuary", "actuarial analyst"],
        ["modèles actuariels", "réserves techniques assurance",
         "Solvabilité II", "tables de mortalité", "pricing assurance vie"],
    ),
    (
        "gestionnaire_sinistres", "banque_assurance",
        "Gestionnaire sinistres",
        ["gestionnaire sinistres", "chargé de sinistres", "liquidateur sinistres",
         "expert sinistres", "claims handler", "claims adjuster"],
        ["déclaration sinistre", "rapport d'expertise", "indemnisation sinistre",
         "garanties assurance IARD", "règlement amiable sinistre"],
    ),
    (
        "charge_clientele_assurance", "banque_assurance",
        "Chargé de clientèle assurance",
        ["chargé de clientèle assurance", "commercial assurance", "agent général assurance",
         "conseiller assurance", "insurance agent", "insurance advisor"],
        ["contrat assurance IARD", "prévoyance collective", "multirisque habitation",
         "commission assurance", "ORIAS", "actes de vente assurance"],
    ),
    (
        "courtier", "banque_assurance",
        "Courtier en assurance",
        ["courtier", "courtier en assurance", "courtier grossiste",
         "courtier santé prévoyance", "insurance broker", "independent broker"],
        ["courtage assurance", "appels d'offres assurance", "comparaison offres assureurs",
         "ORIAS courtier", "mandat de courtage", "négociation tarifs assurance"],
    ),

    # =========================================================
    # FINANCE & COMPTABILITÉ
    # =========================================================

    (
        "comptable", "finance_comptabilite",
        "Comptable",
        ["comptable", "comptable général", "comptable unique",
         "comptable multi-sociétés", "accountant", "bookkeeper"],
        ["grand-livre comptable", "bilan comptable", "déclaration TVA",
         "rapprochement bancaire", "clôture comptable mensuelle", "lettrage"],
    ),
    (
        "controleur_gestion", "finance_comptabilite",
        "Contrôleur de gestion",
        ["contrôleur de gestion", "controller", "analyste gestion",
         "chargé de contrôle de gestion", "management controller",
         "financial controller"],
        ["reporting mensuel financier", "budget prévisionnel", "analyse des écarts budgétaires",
         "tableaux de bord financiers", "SAP CO", "business intelligence finance"],
    ),
    (
        "auditeur", "finance_comptabilite",
        "Auditeur financier",
        ["auditeur", "auditeur financier", "auditeur externe",
         "commissaire aux comptes junior", "auditor", "financial auditor"],
        ["missions d'audit légal", "CAC commissaire aux comptes", "procédures d'audit",
         "liasse fiscale", "contrôle interne", "seuils de signification"],
    ),
    (
        "analyste_financier", "finance_comptabilite",
        "Analyste financier",
        ["analyste financier", "analyste corporate finance", "analyste crédit",
         "analyste M&A", "financial analyst", "investment analyst"],
        ["modélisation financière DCF", "due diligence financière",
         "analyse sectorielle", "Bloomberg terminal", "valorisation entreprise"],
    ),
    (
        "gestionnaire_paie", "finance_comptabilite",
        "Gestionnaire de paie",
        ["gestionnaire de paie", "technicien paie", "responsable paie",
         "chargé de paie", "payroll manager", "payroll specialist"],
        ["bulletin de salaire", "DSN déclaration sociale nominative",
         "cotisations sociales", "SILAE", "ADP Paie", "convention collective paie"],
    ),
    (
        "tresorier", "finance_comptabilite",
        "Trésorier",
        ["trésorier", "responsable trésorerie", "trésorier d'entreprise",
         "gestionnaire trésorerie", "treasurer", "treasury manager"],
        ["prévisions de trésorerie", "cash pooling", "placement financier court terme",
         "lignes de crédit revolving", "Swift bancaire", "netting"],
    ),

    # =========================================================
    # RH & RECRUTEMENT
    # =========================================================

    (
        "responsable_rh", "rh_recrutement",
        "Responsable des ressources humaines",
        ["responsable RH", "DRH", "responsable ressources humaines",
         "HR business partner", "HR manager", "human resources manager"],
        ["GPEC gestion prévisionnelle", "plan de développement des compétences",
         "entretien annuel d'évaluation", "relations sociales CSE",
         "disciplinaire rupture", "SIRH"],
    ),
    (
        "recruteur", "rh_recrutement",
        "Chargé de recrutement",
        ["chargé de recrutement", "recruteur", "talent acquisition",
         "chasseur de têtes", "recruiter", "talent acquisition specialist"],
        ["sourcing LinkedIn Recruiter", "entretien de recrutement structuré",
         "ATS recrutement", "job board Indeed", "onboarding candidats", "pré-qualification"],
    ),
    (
        "gestionnaire_formation", "rh_recrutement",
        "Responsable formation",
        ["responsable formation", "gestionnaire formation", "chargé de formation",
         "learning & development", "L&D manager", "training manager"],
        ["plan de développement des compétences", "OPCO financement", "Qualiopi certification",
         "LMS e-learning", "actions de formation CPF", "bilan de compétences"],
    ),
    (
        "charge_mission_rh", "rh_recrutement",
        "Chargé de mission RH",
        ["chargé de mission RH", "chargé RH", "généraliste RH",
         "HR project manager", "hr generalist"],
        ["projet SIRH", "HRIS déploiement", "réglementation droit du travail",
         "RSE ressources humaines", "égalité professionnelle", "accord collectif"],
    ),

    # =========================================================
    # JURIDIQUE & ADMINISTRATIF
    # =========================================================

    (
        "juriste", "juridique_administratif",
        "Juriste d'entreprise",
        ["juriste", "juriste d'entreprise", "juriste droit des affaires",
         "juriste contrats", "legal counsel", "corporate lawyer"],
        ["rédaction contrats commerciaux", "contentieux judiciaire",
         "droit des sociétés", "compliance RGPD", "actes juridiques", "due diligence juridique"],
    ),
    (
        "avocat", "juridique_administratif",
        "Avocat",
        ["avocat", "avocat d'affaires", "avocat pénaliste",
         "avocat fiscaliste", "attorney", "lawyer"],
        ["barreau inscrit", "plaidoiries", "actes d'avocat",
         "procédure civile", "contentieux judiciaire", "CAPA"],
    ),
    (
        "notaire", "juridique_administratif",
        "Notaire",
        ["notaire", "clerc de notaire", "notaire associé",
         "notaire stagiaire", "notary", "solicitor"],
        ["actes notariés", "droit immobilier notarial", "succession notariale",
         "CRPCEN", "office notarial", "acte authentique"],
    ),
    (
        "secretaire_juridique", "juridique_administratif",
        "Secrétaire juridique",
        ["secrétaire juridique", "assistante juridique", "clerc juridique",
         "assistant cabinet avocat", "legal secretary", "legal assistant"],
        ["gestion agenda avocat", "rédaction actes juridiques",
         "signification tribunal", "archivage dossiers juridiques", "procédure contentieuse"],
    ),
    (
        "assistant_administratif", "juridique_administratif",
        "Assistant administratif",
        ["assistant administratif", "secrétaire", "assistante de direction",
         "agent administratif", "administrative assistant", "office administrator"],
        ["gestion administrative", "classement archivage", "rédaction courriers officiels",
         "accueil téléphonique", "suite bureautique Office", "agenda direction"],
    ),
    (
        "office_manager", "juridique_administratif",
        "Office manager",
        ["office manager", "responsable administratif", "responsable de bureau",
         "coordinateur administratif", "administrative coordinator"],
        ["gestion fournisseurs bureau", "organisation réunions comité",
         "gestion des prestataires", "coordination équipes support",
         "budget frais généraux"],
    ),

    # =========================================================
    # MARKETING & COMMUNICATION
    # =========================================================

    (
        "chef_produit", "marketing_communication",
        "Chef de produit marketing",
        ["chef de produit", "product marketing manager", "chef de marque",
         "responsable produit", "brand manager", "product manager marketing"],
        ["cahier des charges produit", "études de marché", "plan de lancement produit",
         "packaging", "argumentaire produit", "P&L produit"],
    ),
    (
        "marketing_manager", "marketing_communication",
        "Responsable marketing",
        ["responsable marketing", "directeur marketing", "marketing manager",
         "head of marketing", "marketing director"],
        ["plan marketing annuel", "ROI campagnes digitales", "mix marketing 4P",
         "segmentation cible", "budget marketing", "CRM marketing"],
    ),
    (
        "community_manager", "marketing_communication",
        "Community manager",
        ["community manager", "gestionnaire réseaux sociaux", "animateur digital",
         "social media manager", "social media coordinator"],
        ["planning éditorial", "Instagram TikTok", "engagement rate",
         "Canva création contenu", "ads sociaux Meta", "analytics réseaux sociaux"],
    ),
    (
        "graphiste", "marketing_communication",
        "Graphiste",
        ["graphiste", "designer graphique", "directeur artistique",
         "infographiste", "graphic designer", "visual designer"],
        ["Photoshop Illustrator", "InDesign PAO", "charte graphique",
         "Figma maquette", "création visuelle", "identité visuelle"],
    ),
    (
        "webmaster", "marketing_communication",
        "Webmaster",
        ["webmaster", "responsable site web", "webmestre",
         "gestionnaire site internet", "web administrator", "web manager"],
        ["WordPress CMS", "SEO référencement naturel", "Google Analytics 4",
         "maintenance site web", "HTML CSS", "performance web"],
    ),
    (
        "charge_communication", "marketing_communication",
        "Chargé de communication",
        ["chargé de communication", "responsable communication", "attaché communication",
         "communications manager", "communication officer"],
        ["plan de communication annuel", "relations presse", "événementiel entreprise",
         "communiqué de presse", "image de marque", "newsletter"],
    ),
    (
        "relations_presse", "marketing_communication",
        "Attaché de presse",
        ["chargé de relations presse", "attaché de presse", "RP",
         "responsable relations médias", "PR manager", "public relations officer"],
        ["dossier de presse", "retombées presse", "journalistes contacts",
         "revue de presse", "conférence de presse", "communiqué officiel"],
    ),

    # =========================================================
    # INFORMATIQUE & DIGITAL
    # =========================================================

    (
        "developpeur_web", "informatique_digital",
        "Développeur web",
        ["développeur web", "développeur full-stack", "développeur front-end",
         "développeur back-end", "web developer", "software developer",
         "full stack developer"],
        # Short tokens match skills_flat; phrases match bullets/sections
        ["react", "vue", "node.js", "typescript", "api rest", "déploiement web"],
    ),
    (
        "developpeur_mobile", "informatique_digital",
        "Développeur mobile",
        ["développeur mobile", "développeur iOS", "développeur Android",
         "développeur React Native", "mobile developer", "app developer"],
        ["swift", "kotlin", "react native", "flutter", "app store", "sdk mobile"],
    ),
    (
        "devops", "informatique_digital",
        "Ingénieur DevOps",
        ["DevOps", "ingénieur DevOps", "SRE", "ingénieur infrastructure",
         "DevOps engineer", "site reliability engineer"],
        ["kubernetes", "terraform", "helm", "pipeline ci/cd", "infrastructure as code",
         "déploiement continu"],
    ),
    (
        "qa_engineer", "informatique_digital",
        "Ingénieur QA",
        ["ingénieur QA", "testeur logiciel", "QA engineer",
         "ingénieur test", "quality assurance engineer", "test engineer"],
        ["selenium", "pytest", "plan de test", "regression testing", "bdd", "cypress"],
    ),
    (
        "product_manager", "informatique_digital",
        "Product Manager",
        ["product manager", "chef de produit digital", "product owner PO",
         "responsable produit digital", "product owner", "digital product manager"],
        ["roadmap produit", "backlog", "user stories", "okr", "wireframes", "sprint"],
    ),
    (
        "ux_designer", "informatique_digital",
        "UX Designer",
        ["UX designer", "designer UX UI", "designer expérience utilisateur",
         "UI UX designer", "user experience designer", "ux researcher"],
        ["figma", "wireframes", "tests utilisateurs", "parcours utilisateur",
         "design system", "prototypage"],
    ),
    (
        "data_analyst", "informatique_digital",
        "Data Analyst",
        ["data analyst", "analyste de données", "analyste BI",
         "chargé d'études statistiques", "business analyst data",
         "business intelligence analyst", "bi analyst"],
        ["power bi", "tableau", "sql", "reporting data", "dashboard décisionnel",
         "excel avancé"],
    ),
    (
        "data_scientist", "informatique_digital",
        "Data Scientist",
        ["data scientist", "scientifique des données", "ingénieur data science",
         "chercheur en apprentissage automatique", "machine learning scientist",
         "ai researcher"],
        ["scikit-learn", "feature engineering", "validation croisée",
         "statsmodels", "modèle prédictif", "tests a/b"],
    ),
    (
        "ml_engineer", "informatique_digital",
        "Machine Learning Engineer",
        ["machine learning engineer", "ingénieur machine learning",
         "ingénieur IA", "ML engineer", "ingénieur apprentissage automatique",
         "ai engineer", "mlops engineer"],
        # mlflow, kubeflow are specific to ML engineering vs. data science
        ["mlflow", "kubeflow", "pipeline mlops", "déploiement modèle",
         "inférence", "monitoring modèle"],
    ),
    (
        "architecte_logiciel", "informatique_digital",
        "Architecte logiciel",
        ["architecte logiciel", "software architect", "architecte SI",
         "architecte technique", "solutions architect", "technical architect"],
        ["microservices", "design patterns", "scalabilité", "api gateway",
         "architecture hexagonale", "adr décision"],
    ),
    (
        "cybersecurite", "informatique_digital",
        "Ingénieur cybersécurité",
        ["ingénieur cybersécurité", "analyste SOC", "pentester",
         "analyste sécurité informatique", "cybersecurity engineer",
         "security analyst"],
        ["siem", "pentest", "iso 27001", "cve vulnérabilité", "soc analyste",
         "réponse incident"],
    ),

    # =========================================================
    # TOURISME & LOISIRS
    # =========================================================

    (
        "guide_touristique", "tourisme_loisirs",
        "Guide touristique",
        ["guide touristique", "guide-conférencier", "guide du patrimoine",
         "animateur patrimoine", "tour guide", "tour leader"],
        ["visite guidée", "carte nationale de guide-conférencier",
         "médiation culturelle", "patrimoine historique", "commentaire touristique"],
    ),
    (
        "agent_voyages", "tourisme_loisirs",
        "Agent de voyages",
        ["agent de voyages", "conseiller voyages", "consultant voyage",
         "chargé billetterie", "travel agent", "travel consultant"],
        ["réservation voyages GDS Amadeus", "forfait touristique",
         "billetterie aérienne IATA", "offres séjours", "bon de commande voyage"],
    ),
    (
        "animateur_touristique", "tourisme_loisirs",
        "Animateur touristique",
        ["animateur touristique", "animateur village vacances", "animateur club",
         "responsable animations", "holiday animator", "activities coordinator"],
        ["programme animations hôtel", "club vacances", "accueil clientèle touristique",
         "animation sportive loisirs", "BPJEPS animation"],
    ),

    # =========================================================
    # SPORT
    # =========================================================

    (
        "educateur_sportif", "sport",
        "Éducateur sportif",
        ["éducateur sportif", "animateur sportif", "professeur sport",
         "entraîneur sportif", "sports instructor", "sports coach"],
        ["BPJEPS", "DEJEPS", "cours collectifs sportifs",
         "plan d'entraînement", "association sportive", "licence STAPS"],
    ),
    (
        "coach", "sport",
        "Coach sportif",
        ["coach sportif", "coach personnel", "personal trainer",
         "préparateur fitness", "fitness coach", "strength and conditioning coach"],
        ["bilan forme physique", "programme personnalisé fitness",
         "musculation coaching", "BSB brevet sauveteur", "cardio training"],
    ),
    (
        "maitre_nageur", "sport",
        "Maître nageur sauveteur",
        ["maître nageur", "MNS", "maître nageur sauveteur",
         "surveillant de baignade", "lifeguard", "swimming instructor"],
        ["BEESAN", "surveillance bassin", "sauvetage aquatique",
         "BNSSA", "PSE1 premiers secours", "gestion piscine"],
    ),

    # =========================================================
    # CULTURE & MÉDIAS
    # =========================================================

    (
        "journaliste", "culture_medias",
        "Journaliste",
        ["journaliste", "journaliste reporter", "rédacteur en chef",
         "correspondant presse", "journalist", "reporter"],
        ["carte de presse", "rédaction articles presse", "sources journalistiques",
         "reportage terrain", "édition numérique", "ligne éditoriale"],
    ),
    (
        "photographe", "culture_medias",
        "Photographe",
        ["photographe", "reporter photographe", "photographe professionnel",
         "directeur photo", "photographer", "photojournalist"],
        ["prise de vue studio", "retouche Lightroom", "photo reportage",
         "droits à l'image", "Capture One", "commande photo"],
    ),
    (
        "videaste", "culture_medias",
        "Vidéaste",
        ["vidéaste", "cameraman", "réalisateur vidéo",
         "monteur vidéaste", "videographer", "video editor"],
        ["tournage vidéo", "Premiere Pro montage", "After Effects motion",
         "étalonnage DaVinci", "drone vidéo", "format audiovisuel"],
    ),

    # =========================================================
    # ARTISANAT
    # =========================================================

    (
        "boulanger", "artisanat",
        "Boulanger",
        ["boulanger", "boulanger-pâtissier", "artisan boulanger",
         "chef boulanger", "baker", "bread baker"],
        ["pétrissage pâte", "pousse fermentation", "façonnage baguette",
         "levain naturel", "four à sole", "CAP boulangerie"],
    ),
    (
        "patissier", "artisanat",
        "Pâtissier",
        ["pâtissier", "pâtissier-confiseur", "chef pâtissier",
         "artisan pâtissier", "pastry chef", "pastry cook"],
        ["entremets", "ganache chocolat", "tempérage chocolat",
         "décoration gâteaux", "CAP pâtisserie", "pâte feuilletée"],
    ),
    (
        "boucher", "artisanat",
        "Boucher",
        ["boucher", "boucher-charcutier", "artisan boucher",
         "chef boucher", "butcher", "meat cutter"],
        ["désossage viande", "parage filet", "découpe bouchère",
         "présentation rayon boucherie", "CAP boucher", "traçabilité viande"],
    ),
    (
        "fleuriste", "artisanat",
        "Fleuriste",
        ["fleuriste", "artisan fleuriste", "compositeur floral",
         "décorateur floral", "florist", "floral designer"],
        ["composition florale", "bouquet rond deuil", "art floral",
         "plantes vertes d'intérieur", "CAP fleuriste", "décoration événementielle florale"],
    ),
]

# ---------------------------------------------------------------------------
# Build ALL_PROFILES and SECTORS from specs
# ---------------------------------------------------------------------------

ALL_PROFILES: dict[str, SectorProfile] = {}
SECTORS: dict[str, list[str]] = {}

for _spec in _PROFILE_SPECS:
    _id, _sector_key, _title, _aliases, _keywords = _spec
    ALL_PROFILES[_id] = SectorProfile(
        id=_id,
        sector=SECTOR_DISPLAY_NAMES[_sector_key],
        job_title=_title,
        aliases=_aliases,
        detection_keywords=_keywords,
        criteria=[],
    )
    SECTORS.setdefault(_sector_key, []).append(_id)

# display_label → [profile_ids]  (for Phase C dropdown)
PROFILE_BY_SECTOR: dict[str, list[str]] = {
    SECTOR_DISPLAY_NAMES[k]: v for k, v in SECTORS.items()
}

# ---------------------------------------------------------------------------
# Populate criteria for every profile via CriteriaBuilder
# ---------------------------------------------------------------------------

from src.services.criteria_builder import CriteriaBuilder as _CriteriaBuilder  # noqa: E402

_builder = _CriteriaBuilder()
for _sector_key, _pids in SECTORS.items():
    for _pid in _pids:
        ALL_PROFILES[_pid].criteria = _builder.build_for_profile(ALL_PROFILES[_pid], _sector_key)

# ---------------------------------------------------------------------------
# Generic fallback profile — used when confidence < 0.3
# ---------------------------------------------------------------------------

GENERIC_PROFILE = SectorProfile(
    id="non_detecte",
    sector="Générique",
    job_title="Non détecté",
    aliases=["professionnel", "candidat", "demandeur d'emploi", "professional"],
    detection_keywords=[],
    criteria=[
        Criterion(
            id="experience_presente",
            label="Expérience professionnelle présente",
            weight=20,
            required=True,
            detection_fn="has_experience",
            keywords=["expérience", "experience", "poste", "emploi"],
        ),
        Criterion(
            id="formation_presente",
            label="Formation / diplôme présent",
            weight=20,
            required=True,
            detection_fn="has_education",
            keywords=["formation", "diplôme", "étude", "école", "université"],
        ),
        Criterion(
            id="summary_presente",
            label="Accroche / profil présent",
            weight=15,
            required=False,
            detection_fn="has_summary",
            keywords=["profil", "accroche", "présentation", "objectif"],
        ),
        Criterion(
            id="dates_coherentes",
            label="Dates de parcours cohérentes",
            weight=15,
            required=True,
            detection_fn="has_dates",
            keywords=["date", "période", "durée", "depuis"],
        ),
        Criterion(
            id="word_count_300",
            label="CV développé (≥ 300 mots)",
            weight=15,
            required=False,
            detection_fn="has_sufficient_words",
            keywords=[],
        ),
        Criterion(
            id="contact_complet",
            label="Coordonnées de contact complètes",
            weight=15,
            required=False,
            detection_fn="has_contact",
            keywords=["email", "téléphone", "mobile", "adresse"],
        ),
    ],
    esco_occupation_uri=None,
)
