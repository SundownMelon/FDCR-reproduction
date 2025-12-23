# Implementation Plan

- [x] 1. Adapt FedFish Optimizer from fdcr_benign_client.py








  - [x] 1.1 Create FedFish optimizer class by adapting existing implementation


    - Create `Optims/fedfish.py` with FedFish class extending FederatedOptim
    - Port `compute_fisher_information()` method from `fdcr_benign_client.py`
    - Implement `loc_update()` method that trains each client and computes Fisher info
    - Store Fisher info in `self.local_fish_dict[client_index]` for server access
    - Reference: `fdcr_benign_client.py` contains verified Fisher computation logic
    - _Requirements: 4.1, 4.2_
  - [x] 1.2 Write property test for Fisher information computation








    - **Property 4: Per-Round Metric Logging Completeness**
    - **Validates: Requirements 3.1, 3.2**

- [x] 2. Implement DBA (Distributed Backdoor Attack) Module





  - [x] 2.1 Implement DBA trigger partition function


    - Add `get_dba_trigger_partition()` function to `Attack/backdoor/utils.py`
    - Partition full trigger positions evenly across malicious clients
    - Handle edge cases where trigger count is not divisible by client count
    - _Requirements: 1.1, 1.4_
  - [x] 2.2 Write property test for DBA trigger partition






    - **Property 1: DBA Trigger Partition Completeness and Disjointness**
    - **Validates: Requirements 1.1, 1.4**
  - [x] 2.3 Implement DBA backdoor attack function


    - Add `dba_backdoor()` function to `Attack/backdoor/utils.py`
    - Apply only assigned trigger subset to poisoned samples
    - Integrate with existing `backdoor_attack()` function
    - _Requirements: 1.2, 1.3_
  - [x] 2.4 Write property test for DBA trigger application






    - **Property 2: DBA Trigger Application Correctness**
    - **Validates: Requirements 1.2**
  - [x] 2.5 Update configuration to support DBA attack


    - Add `dba_backdoor` option to `utils/cfg.py`
    - Add DBA-specific configuration parameters
    - _Requirements: 1.4_

- [x] 3. Checkpoint - Ensure all tests pass





  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Enhance Server Module with Filtered Ratio Tracking





  - [x] 4.1 Add filtered_ratio computation to OurRandomControl


    - Implement `compute_filtered_ratio()` method
    - Track detection results per round
    - Store filtered_ratio_history for summary
    - _Requirements: 4.3_
  - [x] 4.2 Write property test for filtered_ratio computation






    - **Property 6: Filtered Ratio Computation Correctness**
    - **Validates: Requirements 4.3**
  - [x] 4.3 Add detailed logging to server_update


    - Log benign and malicious client indices each round
    - Log aggregation weights per client
    - _Requirements: 4.1, 4.2_

- [x] 5. Enhance Metrics Logger








  - [x] 5.1 Add filtered_ratio logging to CsvWriter



    - Implement `write_filtered_ratio()` method
    - Implement `write_detection_results()` method
    - _Requirements: 4.3, 4.4_

  - [x] 5.2 Add experiment summary generation

    - Implement `write_summary()` method
    - Compute steady-state ACC/ASR from final window
    - Output mean filtered_ratio and detection accuracy
    - _Requirements: 3.3, 4.4_

- [x] 6. Update Training Loop for New Metrics





  - [x] 6.1 Integrate filtered_ratio logging in training.py


    - Call server's filtered_ratio computation after each round
    - Log detection results via CsvWriter
    - _Requirements: 4.1, 4.2, 4.3_
  - [x] 6.2 Add steady-state metric computation


    - Compute steady-state ACC/ASR from last 10 rounds
    - Output summary at end of training
    - _Requirements: 3.3_

- [x] 7. Checkpoint - Ensure all tests pass





  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Create Batch Experiment Runner





  - [x] 8.1 Create run_experiments.py script


    - Implement experiment configuration combinations
    - Support attack patterns: base_backdoor, dba_backdoor
    - Support alpha values: 0.9, 0.1
    - Support multiple random seeds for reproducibility
    - _Requirements: 5.1, 5.2, 5.4_
  - [x] 8.2 Write property test for experiment reproducibility






    - **Property 7: Experiment Reproducibility**
    - **Validates: Requirements 5.2**
  - [x] 8.3 Implement summary table generation


    - Parse results from all experiment runs
    - Generate markdown table with ACC, ASR, filtered_ratio
    - _Requirements: 5.3_

- [x] 9. Final Integration and Validation







  - [x] 9.1 Update main.py for new attack types
    - Add DBA attack handling in main experiment flow
    - Ensure proper client_type assignment for DBA

    - _Requirements: 1.1, 1.2, 1.3_
  - [x] 9.2 Create example experiment commands

    - Document command-line usage for each configuration
    - Create README section for reproduction experiments
    - _Requirements: 2.4, 5.4_

- [x] 10. Final Checkpoint - Ensure all tests pass





  - Ensure all tests pass, ask the user if questions arise.
