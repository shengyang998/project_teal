//
//  AppRootViewController.swift
//  ProjectTeal
//
//  Created by Soleil Yu on 2025/9/23.
//

import UIKit

class AppRootViewController: UIViewController {
    private let environment: AppEnvironment

    init(environment: AppEnvironment = .shared) {
        self.environment = environment
        super.init(nibName: nil, bundle: nil)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        // Set a default background color
        view.backgroundColor = .systemBackground

        // Embed camera view controller full-screen
        let cameraVC = CameraViewController(environment: environment)
        addChild(cameraVC)
        cameraVC.view.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(cameraVC.view)
        NSLayoutConstraint.activate([
            cameraVC.view.topAnchor.constraint(equalTo: view.topAnchor),
            cameraVC.view.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            cameraVC.view.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            cameraVC.view.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
        cameraVC.didMove(toParent: self)
    }
}
